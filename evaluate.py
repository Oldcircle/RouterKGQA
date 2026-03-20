"""
Unified evaluation script for RouterKGQA.

Combines Stage 3 (query execution + constraint relaxation) and evaluation
(Hits@1, F1) for both WebQSP and CWQ datasets.

Usage:
    python evaluate.py --dataset WebQSP --pred_file path/to/predictions.json --golden_ent
    python evaluate.py --dataset CWQ --pred_file path/to/predictions.json --golden_ent
"""

import argparse
from generation.cwq_evaluate import cwq_evaluate_valid_results
from generation.webqsp_evaluate_offcial import webqsp_evaluate_valid_results
from components.utils import dump_json, load_json
from tqdm import tqdm
from executor.sparql_executor import execute_query_with_odbc, get_2hop_relations_with_odbc_wo_filter
from executor.logic_form_util import lisp_to_sparql
import re
import os
from entity_retrieval import surface_index_memory
import difflib
import itertools
import shutil
import numpy as np
from sentence_transformers import SentenceTransformer

# The embedding model is loaded lazily (after args are parsed)
_sbert_model = None


def _get_sbert_model(model_name):
    """Lazily load and cache the SentenceTransformer model."""
    global _sbert_model
    if _sbert_model is None:
        print(f"Loading embedding model: {model_name}")
        _sbert_model = SentenceTransformer(model_name, trust_remote_code=True)
    return _sbert_model


# ============================================================
# Utility functions
# ============================================================

def is_number(t):
    t = t.replace(" , ", ".")
    t = t.replace(", ", ".")
    t = t.replace(" ,", ".")
    try:
        float(t)
        return True
    except ValueError:
        pass
    try:
        import unicodedata
        unicodedata.numeric(t)
        return True
    except (TypeError, ValueError):
        pass
    return False


def type_checker(token: str):
    """Check the type of a token, e.g. Integer, Float or date.
       Return original token if no type is detected."""
    pattern_year = r"^\d{4}$"
    pattern_year_month = r"^\d{4}-\d{2}$"
    pattern_year_month_date = r"^\d{4}-\d{2}-\d{2}$"
    if re.match(pattern_year, token):
        if int(token) < 3000:  # >= 3000: low possibility to be a year
            token = token + "^^http://www.w3.org/2001/XMLSchema#dateTime"
    elif re.match(pattern_year_month, token):
        token = token + "^^http://www.w3.org/2001/XMLSchema#dateTime"
    elif re.match(pattern_year_month_date, token):
        token = token + "^^http://www.w3.org/2001/XMLSchema#dateTime"
    else:
        return token
    return token


def date_post_process(date_string):
    """
    When querying KB, the KB tends to auto-complete a date.
    e.g.
        - 1996 --> 1996-01-01
        - 1906-04-18 --> 1906-04-18 05:12:00
    """
    pattern_year_month_date = r"^\d{4}-\d{2}-\d{2}$"
    pattern_year_month_date_moment = r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$"

    if re.match(pattern_year_month_date_moment, date_string):
        if date_string.endswith('05:12:00'):
            date_string = date_string.replace('05:12:00', '').strip()
    elif re.match(pattern_year_month_date, date_string):
        if date_string.endswith('-01-01'):
            date_string = date_string.replace('-01-01', '').strip()
    return date_string


# ============================================================
# Constraint stripping / progressive relaxation functions
# ============================================================

SECTION_HEADERS = {
    'main_path': re.compile(r'^main\s*path\s*:$', re.IGNORECASE),
    'entity_restrictions': re.compile(r'^entity\s*restrictions?\s*:$', re.IGNORECASE),
    'numeric_restrictions': re.compile(r'^numeric(al)?\s*restrictions?\s*:$', re.IGNORECASE),
    'string_restrictions': re.compile(r'^string\s*restrictions?\s*:$', re.IGNORECASE),
}


def detect_section(line):
    """Detect whether a line is a section header; return section type or None."""
    stripped = line.strip()
    for sec_type, pattern in SECTION_HEADERS.items():
        if pattern.match(stripped):
            return sec_type
    return None


def strip_constraints(path_str, remove_types):
    """
    Remove specified constraint sections from a path-format string.
    remove_types: set, e.g. {'string_restrictions', 'numeric_restrictions'}
    """
    if not path_str or not remove_types:
        return path_str
    lines = path_str.split('\n')
    result_lines = []
    current_section = None
    skip = False
    for line in lines:
        sec_type = detect_section(line)
        if sec_type is not None:
            current_section = sec_type
            skip = current_section in remove_types
            if not skip:
                result_lines.append(line)
        else:
            if not skip:
                result_lines.append(line)
    return '\n'.join(result_lines).rstrip('\n')


def parse_entity_constraints(path_str):
    """
    Parse path string, splitting Entity Restrictions into individual constraint items.
    Returns: (non_entity_lines, entity_header, entity_constraints)
    """
    lines = path_str.split('\n')
    non_entity_lines = []
    entity_constraints = []
    entity_header = None
    in_entity_section = False
    for line in lines:
        sec_type = detect_section(line)
        if sec_type is not None:
            if sec_type == 'entity_restrictions':
                in_entity_section = True
                entity_header = line
            else:
                in_entity_section = False
                non_entity_lines.append(line)
        else:
            if in_entity_section:
                stripped = line.strip()
                if stripped:
                    entity_constraints.append(line)
            else:
                non_entity_lines.append(line)
    return non_entity_lines, entity_header, entity_constraints


def rebuild_with_entity_subset(non_entity_lines, entity_header, entity_constraints, keep_indices):
    """Rebuild path string with a subset of entity constraints."""
    result_lines = list(non_entity_lines)
    kept = [entity_constraints[i] for i in sorted(keep_indices)]
    if kept and entity_header:
        result_lines.append(entity_header)
        result_lines.extend(kept)
    return '\n'.join(result_lines).rstrip('\n')


def generate_entity_fallback_versions(path_str):
    """
    For a path with string+numeric already removed, generate versions
    that progressively drop entity constraints.
    e.g. with 3 entity constraints (0,1,2):
      drop 1: keep 12, keep 02, keep 01
      drop 2: keep 2, keep 1, keep 0
      (drop all = Q3_main_only, handled by the caller)
    Returns: [(level_name, stripped_path_str), ...]
    """
    from itertools import combinations
    non_entity_lines, entity_header, entity_constraints = parse_entity_constraints(path_str)
    n = len(entity_constraints)
    if n <= 1:
        return []
    versions = []
    for k in range(1, n):  # k = number to drop
        for indices_to_remove in combinations(range(n), k):
            keep_indices = set(range(n)) - set(indices_to_remove)
            removed_str = '_'.join(str(i) for i in indices_to_remove)
            level_name = f"Q2_ent_drop_{removed_str}"
            rebuilt = rebuild_with_entity_subset(
                non_entity_lines, entity_header, entity_constraints, keep_indices
            )
            versions.append((level_name, rebuilt))
    return versions


# ============================================================
# CRP Parser (parse_structured_kbqa_string)
# ============================================================

def parse_structured_kbqa_string(input_str):
    """Parse a structured KBQA path string into its component sections."""
    sections = {
        "main_path": [],
        "main_path_filters": [],
        "entity_restrictions": [],
        "numeric_restrictions": [],
        "string_restrictions": [],
        "other_restrictions": []
    }

    current_section = None
    current_restriction = None  # tracks current restriction dict (numeric/string/main_path_filters)
    triplet_pattern = r'^(\?[\w]+|[mg]\.\w+)\s+-\[(.+?)\]->\s+(\?[\w]+|[mg]\.\w+)$'
    filter_pattern = r'^\s*FILTER\s*\(\s*(.+)\s*\)$'
    order_by_pattern = r'^\s*ORDER BY\s+(ASC|DESC)\s*\(\s*(.+)\s*\)$'
    limit_pattern = r'^\s*LIMIT\s+(\d+)'

    section_patterns = {
        "main_path": r'^(main\s*path:)$',
        "main_path_filters": r'^(main\s*path\s*filters:)$',
        "entity_restrictions": r'^(entity\s*restrictions:)$',
        "numeric_restrictions": r'^(numeric\s*restrictions:)$',
        "string_restrictions": r'^(string\s*restrictions:)$',
        "other_restrictions": r'^(other\s*restrictions:)$'
    }

    for line in input_str.strip().splitlines():
        line = line.strip()
        if not line:
            continue

        # Check if line is a section header
        matched_section = None
        for section_key, pattern in section_patterns.items():
            if re.match(pattern, line, re.IGNORECASE):
                matched_section = section_key
                break

        if matched_section:
            current_section = matched_section
            current_restriction = None  # reset current restriction
            continue

        if not current_section:
            continue

        # Try to parse a triplet
        match = re.match(triplet_pattern, line)
        if match:
            left_entity, relation, right_entity = match.groups()
            triple = (left_entity, relation, right_entity)
            if current_section in ["main_path_filters", "numeric_restrictions", "string_restrictions"]:
                # If previous restriction already has a filter, start a new one
                if current_restriction and "filter" in current_restriction:
                    current_restriction = None
                if not current_restriction:
                    current_restriction = {"path": []}
                    sections[current_section].append(current_restriction)
                current_restriction["path"].append(triple)
            else:
                sections[current_section].append(triple)
            continue

        # Try to parse FILTER
        filter_match = re.match(filter_pattern, line)
        if filter_match and current_section in ["main_path_filters", "numeric_restrictions", "string_restrictions"]:
            filter_content = filter_match.group(1).strip()
            if not current_restriction:
                current_restriction = {"path": [], "filter": f"FILTER ({filter_content})"}
                sections[current_section].append(current_restriction)
            else:
                current_restriction["filter"] = f"FILTER ({filter_content})"
            continue

        # Try to parse ORDER BY
        order_by_match = re.match(order_by_pattern, line)
        if order_by_match and current_section == "numeric_restrictions":
            order_dir = order_by_match.group(1)
            order_var = order_by_match.group(2).strip()
            if current_restriction:
                current_restriction["order_by"] = f"ORDER BY {order_dir}({order_var})"
            continue

        # Try to parse LIMIT
        limit_match = re.match(limit_pattern, line)
        if limit_match and current_section == "numeric_restrictions":
            limit_value = limit_match.group(1)
            if current_restriction:
                current_restriction["limit"] = f"LIMIT {limit_value}"
            continue

    # Extract main_entity: the first entity starting with m. or g.
    main_entity = None
    for left_entity, _, right_entity in sections["main_path"]:
        for entity in [left_entity, right_entity]:
            if entity.startswith('m.') or entity.startswith('g.'):
                main_entity = entity
                break
        if main_entity:
            break

    # target_variable from last triple's object
    target_variable = "?x"
    if sections["main_path"]:
        target_variable = sections["main_path"][-1][2]
    if not main_entity:
        main_entity = "m.03_r3"

    # Clean empty restrictions
    for section in ["main_path_filters", "numeric_restrictions", "string_restrictions"]:
        sections[section] = [r for r in sections[section] if r.get("path") or r.get("filter")]

    result = {
        "main_entity": main_entity,
        "main_path": sections["main_path"],
        "main_path_filters": sections["main_path_filters"],
        "entity_restrictions": sections["entity_restrictions"],
        "numeric_restrictions": sections["numeric_restrictions"],
        "string_restrictions": sections["string_restrictions"],
        "other_restrictions": sections["other_restrictions"],
        "target_variable": target_variable
    }

    return result


# ============================================================
# CRP to SPARQL converter (reconstruct_sparql)
# ============================================================

def create_datatype_query(preliminary_query, order_var):
    """Build a datatype-detection query from a preliminary SPARQL query."""
    lines = preliminary_query.split('\n')
    new_lines = []
    in_select = False
    for line in lines:
        if line.startswith('SELECT'):
            in_select = True
            new_lines.append(f"SELECT DISTINCT (datatype({order_var}) AS ?datatype)")
            continue
        if in_select and line.strip() == 'WHERE {':
            in_select = False
        if line.startswith('ORDER BY'):
            continue
        if line.startswith('LIMIT'):
            new_lines.append('LIMIT 10')
            continue
        new_lines.append(line)
    return '\n'.join(new_lines)


def query_datatype_from_reconstructed(preliminary_query, order_var):
    """Query the Freebase KB to determine the datatype of the ORDER BY variable."""
    datatype_query = create_datatype_query(preliminary_query, order_var)
    print(f"Datatype detection query: {datatype_query}")

    try:
        results = execute_query_with_odbc(datatype_query)
        print(results)

        if results:
            datatype_iri = results[0]
            datatype = re.sub(r'.*#', '', datatype_iri)

            if datatype == "string":
                datatype = "integer"

            print(f"Acquired datatype for {order_var}: {datatype}")
            return datatype

        return None

    except Exception as e:
        print(f"Failed to query datatype for {order_var}: {str(e)}")
        return None


def reconstruct_sparql(parsed_result, dataset="CWQ"):
    """Reconstruct SPARQL queries from parsed CRP result.

    Generates two variants:
      - simple_query: direct triple concatenation
      - optimized_query: with temporal OPTIONAL/BOUND optimization
    """
    main_entity = "ns:" + parsed_result["main_entity"]
    main_path = parsed_result["main_path"]
    entity_restrictions = parsed_result.get("entity_restrictions", [])
    numeric_restrictions = parsed_result.get("numeric_restrictions", [])
    string_restrictions = parsed_result.get("string_restrictions", [])
    other_restrictions = parsed_result.get("other_restrictions", [])
    target_var = "?x"

    # Helper: format a triple
    def format_triple(s, p, o):
        s_full = "ns:" + s if not s.startswith('?') else s
        if o.startswith('?'):
            o_full = o
        elif o.startswith('"') and o.endswith('"'):
            o_full = o
        else:
            o_full = "ns:" + o

        if p.startswith('R[') and p.endswith(']'):
            prop = p[2:-1]
            return f"  {o_full} ns:{prop} {s_full} ."
        else:
            p_full = "ns:" + p if not p.startswith('ns:') else p
            return f"  {s_full} {p_full} {o_full} ."

    # Helper: add numeric suffix to FILTER expressions
    def add_numeric_suffix(filter_str):
        # Handle date format yyyy-mm-dd
        date_pattern = r'(\?[\w]+)\s(<=|>=|<|>|=)\s"(\d{4}-\d{2}-\d{2})"'

        def process_date_filter(match):
            var = match.group(1)
            op = match.group(2)
            date = match.group(3)
            if op == '=':
                return f'regex(str({var}), "^{date}")'
            elif dataset == "CWQ" and op == '>':
                date_parts = date.split('-')
                year = int(date_parts[0])
                new_date = f"{year:04d}-{date_parts[1]}-{date_parts[2]}"
                return f'{var} {op} "{new_date}"^^xsd:dateTime'
            else:
                return f'{var} {op} "{date}"^^xsd:dateTime'
        filter_str = re.sub(date_pattern, process_date_filter, filter_str)

        # Handle year (4-digit number)
        year_pattern = r'(\?[\w]+)\s(<=|>=|<|>|=)\s"(\d{4})"(?!\.\d|\d|\^)'

        def process_year_filter(match):
            var = match.group(1)
            op = match.group(2)
            year = match.group(3)
            if op == '=':
                return f'regex(str({var}), "^{year}")'
            elif dataset == "CWQ" and op == '>':
                new_year = int(year) + 1
                return f'{var} {op} "{new_year:04d}"^^xsd:dateTime'
            else:
                return f'{var} {op} "{year}"^^xsd:dateTime'
        filter_str = re.sub(year_pattern, process_year_filter, filter_str)

        # Handle explicit float (with decimal point)
        float_pattern = r'(\?[\w]+)\s(<=|>=|=|<|>)\s"(-?(?:\d+\.\d+|\.\d+))"(?!\^\^)'
        filter_str = re.sub(float_pattern, r'\1 \2 "\3"^^xsd:float', filter_str)

        # Handle generic numbers (integer, scientific notation, etc.)
        generic_number_pattern = r'(\?[\w]+)\s(<=|>=|<|>|=)\s"(-?[\d.eE+\-]+)"(?!\^\^)'

        def process_generic_number_filter(match):
            var = match.group(1)
            op = match.group(2)
            number = match.group(3)
            is_large_integer = False
            if dataset == "CWQ" and '.' not in number and 'e' not in number.lower():
                digits = number.lstrip('-')
                if len(digits) > 6:
                    is_large_integer = True
                    if number.startswith('-'):
                        prefix = number[:7]
                    else:
                        prefix = number[:6]
                    fuzzy_match = f'regex(str({var}), "^{prefix}")'
                    if op == '=':
                        return fuzzy_match
                    else:
                        return f'{var} {op} "{number}"^^xsd:float'
            if not is_large_integer:
                if op == '=':
                    return f'regex(str({var}), "^{number}")'
                else:
                    return f'{var} {op} "{number}"^^xsd:float'
        filter_str = re.sub(generic_number_pattern, process_generic_number_filter, filter_str)

        return filter_str

    # Helper: convert exact string match to fuzzy match
    def modify_string_filter(filter_str):
        match = re.match(r'FILTER\s*\(str\s*\(\s*(\?\w+)\s*\)\s*=\s*"([^"]*)"\s*\)', filter_str.strip())
        if match:
            var = match.group(1)
            value = match.group(2)
            return f'FILTER(CONTAINS(LCASE(str({var})), LCASE("{value}")))'

        match = re.match(r'FILTER\s*\((\?\w+)\s*=\s*"([^"]*)"\s*\)', filter_str.strip())
        if match:
            var = match.group(1)
            value = match.group(2)
            return f'FILTER(CONTAINS(LCASE({var}), LCASE("{value}")))'

        return filter_str

    # Check for ?c variable
    has_c_variable = False
    all_triples = main_path + entity_restrictions + other_restrictions
    for s, p, o in all_triples:
        if s == '?c' or o == '?c':
            has_c_variable = True
            break

    # Collect ORDER BY and LIMIT info
    preliminary_order_by_clause = None
    limit_clause = None
    order_var = None
    order_dir = None

    for restriction in numeric_restrictions:
        if "order_by" in restriction:
            order_by_str = restriction["order_by"]
            order_var_match = re.search(r'\((\?\w+)\)', order_by_str)
            order_var = order_var_match.group(1) if order_var_match else None
            order_dir = re.search(r'(DESC|ASC)', order_by_str).group(1) if re.search(r'(DESC|ASC)', order_by_str) else "ASC"

            if dataset == "WebQSP":
                order_by_str = re.sub(r'ORDER BY (DESC|ASC)\((\?\w+)\)',
                                     r'ORDER BY \1((\2))', order_by_str)
                preliminary_order_by_clause = order_by_str
            else:  # CWQ
                is_float_type = False
                for s, p, o in restriction.get("path", []):
                    if "population" in p.lower():
                        is_float_type = True
                        break
                if is_float_type:
                    preliminary_order_by_clause = f"ORDER BY {order_dir}(xsd:float({order_var}))"
                else:
                    preliminary_order_by_clause = f"ORDER BY {order_dir}(({order_var}))"
        if "limit" in restriction:
            limit_clause = restriction["limit"]

    # ---- Build simple query ----
    sparql_lines_simple = [
        "PREFIX ns: <http://rdf.freebase.com/ns/>",
        f"SELECT DISTINCT {target_var}",
        "WHERE {"
    ]

    for s, p, o in main_path:
        sparql_lines_simple.append(format_triple(s, p, o))

    for s, p, o in entity_restrictions:
        sparql_lines_simple.append(format_triple(s, p, o))

    for restriction in numeric_restrictions:
        for s, p, o in restriction.get("path", []):
            sparql_lines_simple.append(format_triple(s, p, o))
        if "filter" in restriction:
            filter_str = add_numeric_suffix(restriction["filter"])
            sparql_lines_simple.append(f"  {filter_str}")

    for restriction in string_restrictions:
        for s, p, o in restriction.get("path", []):
            sparql_lines_simple.append(format_triple(s, p, o))
        if "filter" in restriction:
            modified_filter = modify_string_filter(restriction["filter"])
            sparql_lines_simple.append(f" {modified_filter}")

    for s, p, o in other_restrictions:
        sparql_lines_simple.append(format_triple(s, p, o))

    sparql_lines_simple.append(f"  FILTER ({target_var} != {'?c' if has_c_variable else main_entity})")
    sparql_lines_simple.append(f'  FILTER (!isLiteral({target_var}) || lang({target_var}) = "" || langMatches(lang({target_var}), "en"))')

    sparql_lines_simple.append("}")

    if preliminary_order_by_clause:
        sparql_lines_simple.append(preliminary_order_by_clause)
    if limit_clause:
        sparql_lines_simple.append(limit_clause)

    preliminary_simple_query = "\n".join(sparql_lines_simple)

    # For CWQ: query datatype to refine ORDER BY
    order_by_clause = preliminary_order_by_clause
    if dataset == "CWQ" and preliminary_order_by_clause and order_var:
        datatype = query_datatype_from_reconstructed(preliminary_simple_query, order_var)
        if datatype:
            order_by_clause = f"ORDER BY {order_dir}(xsd:{datatype}({order_var}))"
            print(f"Updated ORDER BY with queried datatype: {datatype}")
        else:
            print(f"Datatype query failed, using fallback: {order_by_clause}")

    sparql_lines_simple = preliminary_simple_query.split('\n')
    if preliminary_order_by_clause:
        for i, line in enumerate(sparql_lines_simple):
            if line.startswith('ORDER BY'):
                sparql_lines_simple[i] = order_by_clause

    # ---- Build optimized query (temporal OPTIONAL/BOUND) ----
    sparql_lines_optimized = [
        "PREFIX ns: <http://rdf.freebase.com/ns/>",
        f"SELECT DISTINCT {target_var}",
        "WHERE {"
    ]

    for s, p, o in main_path:
        sparql_lines_optimized.append(format_triple(s, p, o))

    for s, p, o in entity_restrictions:
        sparql_lines_optimized.append(format_triple(s, p, o))

    for restriction in string_restrictions:
        for s, p, o in restriction.get("path", []):
            sparql_lines_optimized.append(format_triple(s, p, o))
        if "filter" in restriction:
            modified_filter = modify_string_filter(restriction["filter"])
            sparql_lines_optimized.append(f" {modified_filter}")

    for s, p, o in other_restrictions:
        sparql_lines_optimized.append(format_triple(s, p, o))

    # Separate date restrictions from other numeric restrictions
    has_date_filter = False
    date_restrictions = []
    other_numeric_restrictions = []
    from_var, to_var, from_prop, to_prop, start_date, end_date = None, None, None, None, None, None

    for restriction in numeric_restrictions:
        filter_str = restriction.get("filter", "")
        path = restriction.get("path", [])

        is_date_restriction = False
        if len(path) > 0:
            prop = path[0][1]
            if re.search(r"(from|start_date)", prop):
                date_match = re.search(r'"(\d{4}-\d{2}-\d{2})"', filter_str)
                if date_match:
                    is_date_restriction = True
                    has_date_filter = True
                    date_value = date_match.group(1)
                    from_var = path[0][2]
                    from_prop = prop
                    start_date = date_value
            elif re.search(r"(to|end_date)", prop):
                date_match = re.search(r'"(\d{4}-\d{2}-\d{2})"', filter_str)
                if date_match:
                    is_date_restriction = True
                    has_date_filter = True
                    date_value = date_match.group(1)
                    to_var = path[0][2]
                    to_prop = prop
                    end_date = date_value

        if is_date_restriction:
            date_restrictions.append(restriction)
        else:
            other_numeric_restrictions.append(restriction)

    # Add temporal OPTIONAL/BOUND filter
    if has_date_filter:
        var = "?y"
        year = start_date[:4] if start_date else end_date[:4]
        sparql_lines_optimized.extend([
            f"  # Temporal logic: only keep records overlapping with year {year}",
            f"  FILTER (",
            f'    (!BOUND(?from) || xsd:dateTime(?from) <= "{start_date if start_date else end_date}"^^xsd:dateTime) &&',
            f'    (!BOUND(?to) || xsd:dateTime(?to) >= "{end_date if end_date else start_date}"^^xsd:dateTime)',
            f"  )",
            f"  OPTIONAL {{ {var} ns:{from_prop} ?from . }}",
            f"  OPTIONAL {{ {var} ns:{to_prop} ?to . }}"
        ])

    # Add remaining numeric restrictions
    for restriction in other_numeric_restrictions:
        for s, p, o in restriction.get("path", []):
            sparql_lines_optimized.append(format_triple(s, p, o))
        if "filter" in restriction:
            filter_str = add_numeric_suffix(restriction["filter"])
            sparql_lines_optimized.append(f"  {filter_str}")

    sparql_lines_optimized.append(f"  FILTER ({target_var} != {'?c' if has_c_variable else main_entity})")
    sparql_lines_optimized.append(f'  FILTER (!isLiteral({target_var}) || lang({target_var}) = "" || langMatches(lang({target_var}), "en"))')

    sparql_lines_optimized.append("}")

    if preliminary_order_by_clause:
        sparql_lines_optimized.append(preliminary_order_by_clause)
    if limit_clause:
        sparql_lines_optimized.append(limit_clause)

    preliminary_optimized_query = "\n".join(sparql_lines_optimized)

    # For CWQ: refine ORDER BY for optimized query too
    if dataset == "CWQ" and preliminary_order_by_clause and order_var:
        datatype = query_datatype_from_reconstructed(preliminary_optimized_query, order_var)
        if datatype:
            order_by_clause = f"ORDER BY {order_dir}(xsd:{datatype}({order_var}))"
            print(f"Updated ORDER BY with queried datatype: {datatype}")
        else:
            print(f"Datatype query failed, using fallback: {order_by_clause}")

    sparql_lines_optimized = preliminary_optimized_query.split('\n')
    if preliminary_order_by_clause:
        for i, line in enumerate(sparql_lines_optimized):
            if line.startswith('ORDER BY'):
                sparql_lines_optimized[i] = order_by_clause

    return {
        "simple_query": "\n".join(sparql_lines_simple),
        "optimized_query": "\n".join(sparql_lines_optimized)
    }


# ============================================================
# Entity/relation denormalization
# ============================================================

def denormalize_s_expr_new(normed_expr, entity_label_map, type_label_map, surface_index):
    """
    Denormalize a structured expression by resolving entity names and relations
    to Freebase MIDs using label maps, embedding similarity, and surface index.
    Returns a list of candidate denormalized expressions.
    """
    model = _get_sbert_model(args.embedding_model)

    def _is_number(s):
        try:
            float(s.replace(',', '.'))
            return True
        except ValueError:
            return False

    def _type_checker(t):
        """Check and process numbers."""
        if _is_number(t):
            return t.replace(" , ", ".")
        return t

    lines = normed_expr.strip().split('\n')
    triplet_pattern = r'(.+?)\s+-\[(.+?)\]->\s*(.+)'

    replacement_map = {}

    for line in lines:
        line = line.strip()
        match = re.match(triplet_pattern, line)
        if match:
            left_entity, relation, right_entity = match.groups()
            left_entity = left_entity.strip()
            relation = relation.strip()
            right_entity = right_entity.strip()

            print(f"Found triplet: {left_entity} -[{relation}]-> {right_entity}")

            # Process left entity (if not a variable)
            if not left_entity.startswith('?'):
                processed_left = None
                if left_entity.lower() in entity_label_map:
                    processed_left = entity_label_map[left_entity.lower()]
                    print(f"Entity match: '{left_entity}' -> '{processed_left}'")
                else:
                    # Try embedding similarity match
                    if len(entity_label_map.keys()) != 0:
                        embeddings_rel = np.array(model.encode([left_entity.lower()], normalize_embeddings=True))
                        embeddings_can = np.array(model.encode(list(entity_label_map.keys()), normalize_embeddings=True))
                        embeddings_st = model.similarity(embeddings_rel, embeddings_can)
                        similarities = embeddings_st.detach().cpu().numpy()
                        merged_list = list(zip([v for _, v in entity_label_map.items()], similarities[0]))
                        sorted_list = sorted(merged_list, key=lambda x: x[1], reverse=True)[0]
                        if sorted_list[1] > 0.1:
                            processed_left = sorted_list[0]
                            print(f"Fuzzy match: '{left_entity}' -> '{processed_left}' [score={sorted_list[1]:.2f}]")

                    # Try surface index
                    if processed_left is None:
                        facc1_cand_entities = surface_index.get_indexrange_entity_el_pro_one_mention(left_entity, top_k=50)
                        if facc1_cand_entities:
                            temp = []
                            for key in list(facc1_cand_entities.keys())[1:]:
                                if facc1_cand_entities[key] >= 0.001:
                                    temp.append(key)
                            if len(temp) > 0:
                                processed_left = [list(facc1_cand_entities.keys())[0]] + temp
                            else:
                                processed_left = list(facc1_cand_entities.keys())[0]
                            print(f"Surface index match: '{left_entity}' -> '{processed_left}'")

                if processed_left is not None:
                    replacement_map[left_entity] = processed_left

            # Process relation
            processed_relation = None
            if relation.lower() in type_label_map:
                processed_relation = type_label_map[relation.lower()]
                print(f"Type match: '{relation}' -> '{processed_relation}'")
            else:
                if ' , ' in relation:
                    if _is_number(relation):
                        processed_relation = relation.replace(" , ", ".").replace(" ,", ".").replace(", ", ".")
                    else:
                        processed_relation = relation.replace(' , ', ',').replace(',', '.').replace(' ', '_')
                        print(f"Relation processed: '{relation}' -> '{processed_relation}'")
                else:
                    if relation.lower() in entity_label_map:
                        processed_relation = entity_label_map[relation.lower()]
                        print(f"Relation entity match: '{relation}' -> '{processed_relation}'")
                    else:
                        # Try embedding similarity
                        if len(entity_label_map.keys()) != 0:
                            embeddings_rel = np.array(model.encode([relation.lower()], normalize_embeddings=True))
                            embeddings_can = np.array(model.encode(list(entity_label_map.keys()), normalize_embeddings=True))
                            embeddings_st = model.similarity(embeddings_rel, embeddings_can)
                            similarities = embeddings_st.detach().cpu().numpy()
                            merged_list = list(zip([v for _, v in entity_label_map.items()], similarities[0]))
                            sorted_list = sorted(merged_list, key=lambda x: x[1], reverse=True)[0]
                            if sorted_list[1] > 0.1:
                                processed_relation = sorted_list[0]
                                print(f"Relation fuzzy match: '{relation}' -> '{processed_relation}' [score={sorted_list[1]:.2f}]")

                        # Try surface index
                        if processed_relation is None:
                            facc1_cand_entities = surface_index.get_indexrange_entity_el_pro_one_mention(relation, top_k=50)
                            if facc1_cand_entities:
                                temp = []
                                for key in list(facc1_cand_entities.keys())[1:]:
                                    if facc1_cand_entities[key] >= 0.001:
                                        temp.append(key)
                                if len(temp) > 0:
                                    processed_relation = [list(facc1_cand_entities.keys())[0]] + temp
                                else:
                                    processed_relation = list(facc1_cand_entities.keys())[0]
                                print(f"Relation surface index match: '{relation}' -> '{processed_relation}'")

            if processed_relation is not None:
                replacement_map[f"[{relation}]"] = f"[{processed_relation}]" if not isinstance(processed_relation, list) else [f"[{r}]" for r in processed_relation]

            # Process right entity (if not a variable)
            if not right_entity.startswith('?'):
                processed_right = None
                if right_entity.lower() in entity_label_map:
                    processed_right = entity_label_map[right_entity.lower()]
                    print(f"Right entity match: '{right_entity}' -> '{processed_right}'")
                else:
                    # Try embedding similarity
                    if len(entity_label_map.keys()) != 0:
                        embeddings_rel = np.array(model.encode([right_entity.lower()], normalize_embeddings=True))
                        embeddings_can = np.array(model.encode(list(entity_label_map.keys()), normalize_embeddings=True))
                        embeddings_st = model.similarity(embeddings_rel, embeddings_can)
                        similarities = embeddings_st.detach().cpu().numpy()
                        merged_list = list(zip([v for _, v in entity_label_map.items()], similarities[0]))
                        sorted_list = sorted(merged_list, key=lambda x: x[1], reverse=True)[0]
                        if sorted_list[1] > 0.2:
                            processed_right = sorted_list[0]
                            print(f"Right fuzzy match: '{right_entity}' -> '{processed_right}' [score={sorted_list[1]:.2f}]")

                    # Try surface index
                    if processed_right is None:
                        facc1_cand_entities = surface_index.get_indexrange_entity_el_pro_one_mention(right_entity, top_k=50)
                        if facc1_cand_entities:
                            temp = []
                            for key in list(facc1_cand_entities.keys())[1:]:
                                if facc1_cand_entities[key] >= 0.001:
                                    temp.append(key)
                            if len(temp) > 0:
                                processed_right = [list(facc1_cand_entities.keys())[0]] + temp
                            else:
                                processed_right = list(facc1_cand_entities.keys())[0]
                            print(f"Right surface index match: '{right_entity}' -> '{processed_right}'")

                if processed_right is not None:
                    replacement_map[right_entity] = processed_right

    print(f"Replacement map: {replacement_map}")

    # Generate all possible combinations
    replacement_keys = list(replacement_map.keys())
    replacement_values = [replacement_map[key] if isinstance(replacement_map[key], list) else [replacement_map[key]] for key in replacement_keys]

    combinations = [list(comb) for comb in itertools.islice(itertools.product(*replacement_values), 10000)]

    # Generate final expressions
    exprs = []
    for combo in combinations:
        expr_copy = normed_expr
        for i, key in enumerate(replacement_keys):
            expr_copy = expr_copy.replace(key, combo[i])
        exprs.append(expr_copy)

    print(f"\nGenerated {len(exprs)} expression combinations")
    for example in exprs[:3]:
        print("Example:", example)
    if len(exprs) > 3:
        print("...")

    return exprs


# ============================================================
# Main execution function
# ============================================================

def execute_normed_s_expr_from_label_maps_main(normed_expr,
                                                entity_label_map,
                                                type_label_map,
                                                surface_index):
    """Execute with only main path (all constraints stripped for CRP parser)."""
    try:
        denorm_sexprs = denormalize_s_expr_new(
            normed_expr,
            entity_label_map,
            type_label_map,
            surface_index
        )
    except Exception as e:
        print(f"Failed to denormalize expression: {e}")
        return 'null', []

    query_exprs = denorm_sexprs

    for query_expr in query_exprs[:500]:
        try:
            print("Denormalized Expression:\n", query_expr)

            parsed_result = parse_structured_kbqa_string(query_expr)
            # Only keep main path, strip all constraints
            parsed_result = {
                "main_entity": parsed_result.get("main_entity", ""),
                "main_path": parsed_result.get("main_path", []),
                "main_path_filters": parsed_result.get("main_path_filters", []),
                "entity_restrictions": [],
                "numeric_restrictions": [],
                "string_restrictions": [],
                "other_restrictions": [],
                "target_variable": ""
            }
            sparql_queries = reconstruct_sparql(parsed_result, dataset=args.dataset)
            simple_query = sparql_queries["simple_query"]
            optimized_query = sparql_queries["optimized_query"]

            print("\nReconstructed SPARQL (Simple Query):")
            print(simple_query)
            print("\nReconstructed SPARQL (Optimized Query):")
            print(optimized_query)

            denotation = []

            if "BOUND" in optimized_query:
                try:
                    denotation = execute_query_with_odbc(optimized_query)
                    denotation = [res.replace("http://rdf.freebase.com/ns/", '') for res in denotation]
                    if denotation:
                        print(f"Optimized Query returned {len(denotation)} results.")
                        return query_expr, denotation
                except Exception as e:
                    print(f"Optimized query failed: {e}")
                    denotation = []

            if not denotation:
                try:
                    denotation = execute_query_with_odbc(simple_query)
                    denotation = [res.replace("http://rdf.freebase.com/ns/", '') for res in denotation]
                    if denotation:
                        print(f"Simple Query returned {len(denotation)} results.")
                        return query_expr, denotation
                    else:
                        print("Simple query returned no results.")
                except Exception as e:
                    print(f"Simple query execution failed: {e}")
                    denotation = []

        except Exception as e:
            print(f"Failed processing expression: {e}")

    print("All queries failed. Returning first expression with empty result.")
    return query_exprs[0], []


def execute_normed_s_expr_from_label_maps(normed_expr,
                                          entity_label_map,
                                          type_label_map,
                                          surface_index):
    """Execute denormalized expressions against the KB."""
    try:
        denorm_sexprs = denormalize_s_expr_new(
            normed_expr,
            entity_label_map,
            type_label_map,
            surface_index
        )
    except Exception as e:
        print(f"Failed to denormalize expression: {e}")
        return 'null', []

    query_exprs = denorm_sexprs

    for query_expr in query_exprs[:500]:
        try:
            print("Denormalized Expression:\n", query_expr)

            parsed_result = parse_structured_kbqa_string(query_expr)

            sparql_queries = reconstruct_sparql(parsed_result, dataset=args.dataset)
            simple_query = sparql_queries["simple_query"]
            optimized_query = sparql_queries["optimized_query"]

            print("\nReconstructed SPARQL (Simple Query):")
            print(simple_query)
            print("\nReconstructed SPARQL (Optimized Query):")
            print(optimized_query)

            denotation = []

            if "BOUND" in optimized_query:
                try:
                    denotation = execute_query_with_odbc(optimized_query)
                    denotation = [res.replace("http://rdf.freebase.com/ns/", '') for res in denotation]
                    if denotation:
                        print(f"Optimized Query returned {len(denotation)} results.")
                        return query_expr, denotation
                except Exception as e:
                    print(f"Optimized query failed: {e}")
                    denotation = []

            if not denotation:
                try:
                    denotation = execute_query_with_odbc(simple_query)
                    denotation = [res.replace("http://rdf.freebase.com/ns/", '') for res in denotation]
                    if denotation:
                        print(f"Simple Query returned {len(denotation)} results.")
                        return query_expr, denotation
                    else:
                        print("Simple query returned no results.")
                except Exception as e:
                    print(f"Simple query execution failed: {e}")
                    denotation = []

        except Exception as e:
            print(f"Failed processing expression: {e}")

    print("All queries failed. Returning first expression with empty result.")
    return query_exprs[0], []


# ============================================================
# Relation replacement fallback (constraint-only)
# ============================================================

def _get_constraint_region_start(text):
    """Find the start offset of the constraint region (first non-Main Path section header).
    Returns offset in text, or len(text) if no constraints exist."""
    lines = text.split('\n')
    offset = 0
    current_section = None
    for line in lines:
        sec_type = detect_section(line)
        if sec_type is not None:
            if sec_type != 'main_path' and current_section == 'main_path':
                return offset
            current_section = sec_type
        offset += len(line) + 1  # +1 for '\n'
    return len(text)


def try_relation(d, gen_feat, constraint_only=True):
    """Replace relations in denormalized expression with candidate relations
    using embedding similarity. Only replaces constraints, keeps main path intact."""
    model = _get_sbert_model(args.embedding_model)
    r_pattern = r'\[R\[([^\]]+)\]\]'
    bracket_pattern = r'\[([^\]]+)\]'

    # Determine constraint region start
    if constraint_only:
        constraint_start = _get_constraint_region_start(d)
        print(f"Constraint region start: {constraint_start}/{len(d)}")
        if constraint_start >= len(d):
            print("No constraint section, skipping relation replacement")
            return d, []
    else:
        constraint_start = 0

    # Extract relations from constraint region only
    constraint_text = d[constraint_start:]
    rel_list = []

    r_matches = re.findall(r_pattern, constraint_text)
    for match in r_matches:
        if match not in rel_list:
            rel_list.append(match)

    temp_ct = re.sub(r_pattern, 'TEMP_PLACEHOLDER', constraint_text)
    bracket_matches = re.findall(bracket_pattern, temp_ct)
    for match in bracket_matches:
        if match != 'TEMP_PLACEHOLDER' and match not in rel_list:
            rel_list.append(match)

    print(f"Original data: {d}")
    print(f"Constraint-region relations: {rel_list}")

    cand_rels = []
    if gen_feat.get('relation_retrieval') and gen_feat['relation_retrieval'].get('question_relations'):
        cand_rels = [rel['relation'] for rel in gen_feat['relation_retrieval']['question_relations']['top_relations']]

    if len(cand_rels) == 0 or len(rel_list) == 0:
        return d, []

    embeddings_rel = np.array(model.encode(rel_list, normalize_embeddings=True))
    embeddings_can = np.array(model.encode(cand_rels, normalize_embeddings=True))
    embeddings_st = model.similarity(embeddings_rel, embeddings_can)
    similarities = embeddings_st.detach().cpu().numpy()

    print("Relations being processed:", rel_list)
    print("Similarity matrix:", similarities)

    change = dict()
    for i, rel in enumerate(rel_list):
        merged_list = list(zip(cand_rels, similarities[i]))
        sorted_list = sorted(merged_list, key=lambda x: x[1], reverse=True)
        change_rel = []
        for s in sorted_list:
            if s[1] > 0.01:
                change_rel.append(s[0])
        change[rel] = change_rel[:15]

    print("Relation replacement map:", change)

    # Locate relation positions in full text, but only replace those in constraint region
    original_text = d
    relation_positions = []

    for match in re.finditer(r_pattern, original_text):
        if match.start() < constraint_start:
            continue  # skip main path region
        relation = match.group(1)
        if relation in change and change[relation]:
            relation_positions.append({
                'start': match.start(),
                'end': match.end(),
                'original': match.group(0),
                'relation': relation,
                'format': 'R'
            })

    r_matches_positions = [(m.start(), m.end()) for m in re.finditer(r_pattern, original_text)]

    for match in re.finditer(bracket_pattern, original_text):
        if match.start() < constraint_start:
            continue  # skip main path region
        is_inside_r = False
        for r_start, r_end in r_matches_positions:
            if r_start <= match.start() < r_end:
                is_inside_r = True
                break

        if not is_inside_r:
            relation = match.group(1)
            if relation in change and change[relation]:
                relation_positions.append({
                    'start': match.start(),
                    'end': match.end(),
                    'original': match.group(0),
                    'relation': relation,
                    'format': 'bracket'
                })

    relation_positions.sort(key=lambda x: x['start'], reverse=True)

    if not relation_positions:
        return original_text, []

    replacement_options = []
    for pos in relation_positions:
        relation = pos['relation']
        if relation in change:
            replacement_options.append(change[relation])
        else:
            replacement_options.append([relation])

    combinations_list = list(itertools.islice(
        itertools.product(*replacement_options), 4000
    ))

    result_texts = []
    for combination in combinations_list:
        current_text = original_text
        for i, pos in enumerate(relation_positions):
            replacement_relation = combination[i]
            if pos['format'] == 'R':
                new_text = f"[R[{replacement_relation}]]"
            else:
                new_text = f"[{replacement_relation}]"
            current_text = current_text[:pos['start']] + new_text + current_text[pos['end']:]
        result_texts.append(current_text)

    result_texts = list(set(result_texts))

    print(f"Generated {len(result_texts)} constraint-relation replacement combinations (main path preserved)")
    for query_expr in result_texts[:300]:
        try:
            print("Denormalized Expression:\n", query_expr)

            parsed_result = parse_structured_kbqa_string(query_expr)
            sparql_queries = reconstruct_sparql(parsed_result, dataset=args.dataset)
            simple_query = sparql_queries["simple_query"]
            optimized_query = sparql_queries["optimized_query"]

            print("\nReconstructed SPARQL (Simple Query):")
            print(simple_query)
            print("\nReconstructed SPARQL (Optimized Query):")
            print(optimized_query)

            denotation = []

            if "BOUND" in optimized_query:
                try:
                    denotation = execute_query_with_odbc(optimized_query)
                    denotation = [res.replace("http://rdf.freebase.com/ns/", '') for res in denotation]
                    if denotation:
                        print(f"Optimized Query returned {len(denotation)} results.")
                        return query_expr, denotation
                except Exception as e:
                    print(f"Optimized query failed: {e}")
                    denotation = []

            if not denotation:
                try:
                    denotation = execute_query_with_odbc(simple_query)
                    denotation = [res.replace("http://rdf.freebase.com/ns/", '') for res in denotation]
                    if denotation:
                        print(f"Simple Query returned {len(denotation)} results.")
                        return query_expr, denotation
                    else:
                        print("Simple query returned no results.")
                except Exception as e:
                    print(f"Simple query execution failed: {e}")
                    denotation = []

        except Exception as e:
            print(f"Failed processing expression: {e}")

    print("All queries failed. Returning first expression with empty result.")
    return result_texts[0], []


def execute_normed_s_expr_from_label_maps_rel(normed_expr,
                                              entity_label_map,
                                              type_label_map,
                                              surface_index,
                                              gen_feat):
    """Denormalize first, then try relation replacement for each candidate."""
    try:
        denorm_sexprs = denormalize_s_expr_new(normed_expr,
                                               entity_label_map,
                                               type_label_map,
                                               surface_index)
    except:
        return 'null', []

    query_exprs = denorm_sexprs

    denotation = []
    query_expr = query_exprs[0] if query_exprs else 'null'
    for d in tqdm(denorm_sexprs[:30], desc='try_relation'):
        query_expr, denotation = try_relation(d, gen_feat)
        if len(denotation) != 0:
            break

    if len(denotation) == 0:
        query_expr = query_exprs[0]

    return query_expr, denotation


# ============================================================
# Fallback cascade
# ============================================================

# Fallback levels:
# Q0_full:       Full prediction (main path + entity + numeric + string)
# Q1_no_string:  Remove string restrictions
# Q2_no_str_num: Remove string + numeric restrictions
# Q2_ent_drop_X: Remove string + numeric + progressively drop entity constraint subsets
# Q3_main_only:  Remove all constraints (main path only)
#
# Execution order:
# Phase 1: Try full query (Q0_full) for all predictions in order
# Phase 2: If Phase 1 fails, for each prediction run the full fallback cascade:
#          Q1_no_string -> Q2_no_str_num -> Q2_ent_drop_X -> Q3_main_only
#          Only after all levels fail for prediction 1, try prediction 2, etc.


def _run_fallback_cascade_for_prediction(p, entity_label_map, train_type_map,
                                          surface_index, qid, rank, dataset):
    """Run the constraint fallback cascade for a single prediction.
    Returns (lf, answers, hit_level) or (None, None, None) if all fail.
    """

    # Step 1: Q1_no_string, Q2_no_str_num
    for level_name, remove_types in [
        ("Q1_no_string",  {'string_restrictions'}),
        ("Q2_no_str_num", {'string_restrictions', 'numeric_restrictions'}),
    ]:
        print(f"\n{'='*60}")
        print(f"  Fallback {level_name} rank={rank} (qid={qid})")
        print(f"{'='*60}")

        stripped_p = strip_constraints(p, remove_types)
        lf, answers = execute_normed_s_expr_from_label_maps(
            stripped_p, entity_label_map, train_type_map, surface_index
        )
        answers = [date_post_process(ans) for ans in list(answers)]
        if answers:
            return lf, answers, level_name

    # Step 2: Q2_ent_drop_X (progressively drop entity constraints)
    base_p = strip_constraints(p, {'string_restrictions', 'numeric_restrictions'})
    entity_versions = generate_entity_fallback_versions(base_p)

    for level_name, stripped_p in entity_versions:
        print(f"\n{'='*60}")
        print(f"  Fallback {level_name} rank={rank} (qid={qid})")
        print(f"{'='*60}")

        lf, answers = execute_normed_s_expr_from_label_maps(
            stripped_p, entity_label_map, train_type_map, surface_index
        )
        answers = [date_post_process(ans) for ans in list(answers)]
        if answers:
            return lf, answers, level_name

    # Step 3: Q3_main_only
    print(f"\n{'='*60}")
    print(f"  Fallback Q3_main_only rank={rank} (qid={qid})")
    print(f"{'='*60}")

    if dataset == "WebQSP":
        # WebQSP uses execute_normed_s_expr_from_label_maps_main (strips in parser)
        lf, answers = execute_normed_s_expr_from_label_maps_main(
            p, entity_label_map, train_type_map, surface_index
        )
    else:
        # CWQ strips all constraints via strip_constraints
        stripped_p = strip_constraints(p, {'string_restrictions', 'numeric_restrictions', 'entity_restrictions'})
        lf, answers = execute_normed_s_expr_from_label_maps(
            stripped_p, entity_label_map, train_type_map, surface_index
        )
    answers = [date_post_process(ans) for ans in list(answers)]
    if answers:
        return lf, answers, "Q3_main_only"

    return None, None, None


# ============================================================
# Main evaluation loop
# ============================================================

def aggressive_top_k_eval_new(split, predict_file, dataset):
    """Run top-k predictions with progressive fallback strategy.

    Phase 1: Try full query (Q0_full) for all predictions in order.
    Phase 2: If all fail, for each prediction run the full fallback cascade.
    """
    if dataset == "CWQ":
        train_gen_dataset = load_json('data/CWQ/generation/merged/CWQ_train.json')
        test_gen_dataset = load_json('data/CWQ/generation/merged/CWQ_test.json')
        dev_gen_dataset = None
    elif dataset == "WebQSP":
        train_gen_dataset = load_json('data/WebQSP/generation/merged/WebQSP_train.json')
        test_gen_dataset = load_json('data/WebQSP/generation/merged/WebQSP_test.json')
        dev_gen_dataset = None

    predictions = load_json(predict_file)
    print(os.path.dirname(predict_file))
    dirname = os.path.dirname(predict_file)
    filename = os.path.basename(predict_file)
    if split == 'dev':
        gen_dataset = dev_gen_dataset
    elif split == 'train':
        gen_dataset = train_gen_dataset
    else:
        gen_dataset = test_gen_dataset
    if dataset == "CWQ":
        train_type_map = load_json("data/CWQ/generation/label_maps/CWQ_train_type_label_map.json")
        train_type_map = {l.lower(): t for t, l in train_type_map.items()}
    elif dataset == "WebQSP":
        train_type_map = load_json("data/WebQSP/generation/label_maps/WebQSP_train_type_label_map.json")
        train_type_map = {l.lower(): t for t, l in train_type_map.items()}

    surface_index = surface_index_memory.EntitySurfaceIndexMemory(
        "data/common_data/facc1/entity_list_file_freebase_complete_all_mention",
        "data/common_data/facc1/surface_map_file_freebase_complete_all_mention",
        "data/common_data/facc1/freebase_complete_all_mention")

    ex_cnt = 0
    top_hit = 0
    lines = []
    official_lines = []
    failed_preds = []
    gen_executable_cnt = 0
    final_executable_cnt = 0
    level_hit_cnt = {}
    processed = 0

    for (pred, gen_feat) in tqdm(zip(predictions, gen_dataset), total=len(gen_dataset), desc=f'Evaluating {split}'):

        qid = gen_feat['ID']

        if args.golden_ent:
            entity_label_map = {v.lower(): k for k, v in list(gen_feat['gold_entity_map'].items())}
        else:
            entity_label_map = {}

        executable_index = None
        final_lf = None
        final_answers = None
        final_denormed_pred = []
        hit_level = None

        # ========== Phase 1: Full query execution (all predictions) ==========
        print(f"\n{'='*60}")
        print(f"Phase 1: Q0_full (qid={qid})")
        print(f"{'='*60}")

        denormed_pred = []
        for rank, p in enumerate(pred['predictions']):
            lf, answers = execute_normed_s_expr_from_label_maps(
                p,
                entity_label_map,
                train_type_map,
                surface_index
            )
            answers = [date_post_process(ans) for ans in list(answers)]

            denormed_pred.append(lf)
            if rank == 0 and lf.lower() == gen_feat['sexpr'].lower():
                ex_cnt += 1

            if answers:
                executable_index = rank
                final_lf = lf
                final_answers = answers
                final_denormed_pred = denormed_pred
                hit_level = "Q0_full"
                if rank == 0:
                    top_hit += 1
                break

        if executable_index is not None:
            gen_executable_cnt += 1

        # ========== Phase 2: Per-prediction fallback cascade ==========
        if executable_index is None:
            print(f"\n{'='*60}")
            print(f"Phase 2: Per-prediction fallback cascade (qid={qid})")
            print(f"{'='*60}")

            for rank, p in enumerate(pred['predictions']):
                lf, answers, level = _run_fallback_cascade_for_prediction(
                    p, entity_label_map, train_type_map,
                    surface_index, qid, rank, dataset
                )

                if answers:
                    executable_index = rank
                    final_lf = lf
                    final_answers = answers
                    final_denormed_pred = [lf]
                    hit_level = level
                    if rank == 0:
                        top_hit += 1
                    break

        # Record results
        if executable_index is not None:
            final_executable_cnt += 1
            level_hit_cnt[hit_level] = level_hit_cnt.get(hit_level, 0) + 1
            lines.append({
                'qid': qid,
                'execute_index': executable_index,
                'fallback_level': hit_level,
                'logical_form': final_lf,
                'answer': final_answers,
                'gt_sexpr': gen_feat['sexpr'],
                'gt_normed_sexpr': pred['gen_label'],
                'pred': pred,
                'denormed_pred': final_denormed_pred
            })
            official_lines.append({
                "QuestionId": qid,
                "Answers": final_answers
            })
        else:
            failed_preds.append({
                'qid': qid,
                'gt_sexpr': gen_feat['sexpr'],
                'gt_normed_sexpr': pred['gen_label'],
                'pred': pred,
                'denormed_pred': final_denormed_pred
            })

        processed += 1
        if processed % 100 == 0:
            print(f'Processed:{processed}, executable:{final_executable_cnt}')
            for lname in sorted(level_hit_cnt.keys()):
                print(f'  {lname}: {level_hit_cnt[lname]}')

    print('STR Match', ex_cnt / len(predictions))
    print('TOP 1 Executable', top_hit / len(predictions))
    print('Gen Executable', gen_executable_cnt / len(predictions))
    print('Final Executable', final_executable_cnt / len(predictions))
    print('--- Fallback Level Statistics ---')
    for lname in sorted(level_hit_cnt.keys()):
        cnt = level_hit_cnt[lname]
        print(f'  {lname}: {cnt} ({cnt/len(predictions)*100:.2f}%)')

    result_file = os.path.join(dirname, f'{filename}_gen_sexpr_results.json')
    official_results_file = os.path.join(dirname, f'{filename}_gen_sexpr_results_official_format.json')
    dump_json(lines, result_file, indent=4)
    dump_json(official_lines, official_results_file, indent=4)
    dump_json(failed_preds, os.path.join(dirname, f'{filename}_gen_failed_results.json'), indent=4)
    dump_json({
        'STR Match': ex_cnt / len(predictions),
        'TOP 1 Executable': top_hit / len(predictions),
        'Gen Executable': gen_executable_cnt / len(predictions),
        'Final Executable': final_executable_cnt / len(predictions),
        'Fallback Level Stats': dict(sorted(level_hit_cnt.items()))
    }, os.path.join(dirname, f'{filename}_statistics.json'), indent=4)

    # Run final evaluation (Hits@1, F1)
    if dataset == "CWQ":
        args.pred_file = result_file
        cwq_evaluate_valid_results(args)
    else:
        args.pred_file = official_results_file
        webqsp_evaluate_valid_results(args)


# ============================================================
# Argument parsing and entry point
# ============================================================

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Unified evaluation for RouterKGQA (WebQSP / CWQ)"
    )
    parser.add_argument('--dataset', default='WebQSP', type=str,
                        choices=['WebQSP', 'CWQ'],
                        help='Dataset type: WebQSP or CWQ')
    parser.add_argument('--pred_file', required=True, type=str,
                        help='Path to top-k prediction file')
    parser.add_argument('--split', default='test', type=str,
                        help='Split to operate on: test, dev, or train')
    parser.add_argument('--golden_ent', default=True, action='store_true',
                        help='Use golden entities (default: True)')
    parser.add_argument('--no_golden_ent', dest='golden_ent', action='store_false',
                        help='Disable golden entity mode')
    parser.add_argument('--embedding_model', default='nomic-ai/nomic-embed-text-v1', type=str,
                        help='SBERT model for embedding similarity (default: nomic-ai/nomic-embed-text-v1)')
    parser.add_argument('--beam_size', default=50, type=int,
                        help='Beam size for top-k predictions')

    parsed_args = parser.parse_args()

    print(f'Dataset: {parsed_args.dataset}, Split: {parsed_args.split}, '
          f'Pred file: {parsed_args.pred_file}, Golden entities: {parsed_args.golden_ent}')
    return parsed_args


if __name__ == '__main__':
    args = _parse_args()

    if args.golden_ent:
        new_dir_path = os.path.join(os.path.dirname(args.pred_file), 'golden_ent_predict')
        if not os.path.exists(new_dir_path):
            os.makedirs(new_dir_path)
        new_dir_name = os.path.join(new_dir_path, args.pred_file.split('/')[-1])
        shutil.copyfile(args.pred_file, new_dir_name)
        args.pred_file = new_dir_name

    aggressive_top_k_eval_new(args.split, args.pred_file, args.dataset)

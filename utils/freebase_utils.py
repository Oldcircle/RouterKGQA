"""Freebase knowledge graph utility functions.

Provides SPARQL query execution, entity/relation search, CRP parsing,
SPARQL reconstruction, embedding computation, and LLM-based path selection
for Freebase-backed KGQA pipelines.
"""

import os
import re
import json
import numpy as np

from SPARQLWrapper import SPARQLWrapper, JSON

# ---------------------------------------------------------------------------
# SPARQL endpoint -- configurable via environment variable
# ---------------------------------------------------------------------------
SPARQLPATH = os.environ.get("SPARQL_ENDPOINT", "http://localhost:3001/sparql")

# ---------------------------------------------------------------------------
# SPARQL query templates
# ---------------------------------------------------------------------------

sparql_all_relations = """
PREFIX fb: <http://rdf.freebase.com/ns/>
SELECT DISTINCT ?relation WHERE {
  {
    fb:%s ?relation ?object .
  } UNION {
    ?subject ?relation fb:%s .
  }
  FILTER(STRSTARTS(STR(?relation), "http://rdf.freebase.com/ns/"))
}
"""

sparql_head_relations = """
PREFIX ns: <http://rdf.freebase.com/ns/>
SELECT ?relation WHERE {
  ns:%s ?relation ?x .
}
"""

sparql_tail_relations = """
PREFIX ns: <http://rdf.freebase.com/ns/>
SELECT ?relation WHERE {
  ?x ?relation ns:%s .
}
"""

sparql_tail_entities_extract = (
    "PREFIX ns: <http://rdf.freebase.com/ns/>\n"
    "SELECT ?tailEntity\n"
    "WHERE {\n"
    "ns:%s ns:%s ?tailEntity .\n"
    "}"
)

sparql_head_entities_extract = (
    "PREFIX ns: <http://rdf.freebase.com/ns/>\n"
    "SELECT ?tailEntity\n"
    "WHERE {\n"
    "?tailEntity ns:%s ns:%s  .\n"
    "}"
)

sparql_attribute_extract = """PREFIX ns: <http://rdf.freebase.com/ns/>
SELECT ?value
WHERE {
  ns:%s ns:%s ?value .
  FILTER (
    isLiteral(?value) &&
    (lang(?value) = 'en' || lang(?value) = '')
  )
}"""

sparql_id = (
    "PREFIX ns: <http://rdf.freebase.com/ns/>\n"
    "SELECT DISTINCT ?tailEntity\n"
    "WHERE {\n"
    "  {\n"
    "    ?entity ns:type.object.name ?tailEntity .\n"
    "    FILTER(?entity = ns:%s)\n"
    "  }\n"
    "  UNION\n"
    "  {\n"
    "    ?entity <http://www.w3.org/2002/07/owl#sameAs> ?tailEntity .\n"
    "    FILTER(?entity = ns:%s)\n"
    "  }\n"
    "}"
)

# ---------------------------------------------------------------------------
# Low-level SPARQL helpers
# ---------------------------------------------------------------------------


def execute_sparql(sparql_query):
    """Execute a SPARQL query against the configured Virtuoso endpoint."""
    sparql = SPARQLWrapper(SPARQLPATH)
    sparql.setQuery(sparql_query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    return results["results"]["bindings"]


def replace_relation_prefix(relations):
    """Strip the Freebase namespace prefix from relation URIs."""
    return [
        relation["relation"]["value"].replace("http://rdf.freebase.com/ns/", "")
        for relation in relations
    ]


def replace_entities_prefix(entities):
    """Strip the Freebase namespace prefix from entity URIs."""
    return [
        entity["tailEntity"]["value"].replace("http://rdf.freebase.com/ns/", "")
        for entity in entities
    ]


def extract_sparql_entities(query_results, variable_name="x"):
    """Extract entity IDs from SPARQL query results.

    Args:
        query_results: SPARQL result bindings list.
        variable_name: Variable name to extract (default ``'x'``).

    Returns:
        List of entity ID strings with the Freebase prefix removed.
    """
    entities = []
    for result in query_results:
        if variable_name in result and "value" in result[variable_name]:
            entity_uri = result[variable_name]["value"]
            entity_id = entity_uri.replace("http://rdf.freebase.com/ns/", "")
            entities.append(entity_id)
    return entities


# ---------------------------------------------------------------------------
# Entity / attribute search
# ---------------------------------------------------------------------------


def entity_search(entity, relation, head=True):
    """Search for entities connected via *relation* to *entity*.

    Args:
        entity: Freebase entity ID (e.g. ``'m.03_r3'``).
        relation: Relation string.
        head: If ``True`` treat *entity* as subject; otherwise as object.

    Returns:
        List of connected entity IDs starting with ``m.`` or ``g.``.
    """
    if head:
        query = sparql_tail_entities_extract % (entity, relation)
    else:
        query = sparql_head_entities_extract % (relation, entity)
    entities = execute_sparql(query)
    entity_ids = replace_entities_prefix(entities)
    return [eid for eid in entity_ids if eid.startswith("m.") or eid.startswith("g.")]


def attribute_search(entity, attribute):
    """Retrieve literal attribute values for an entity."""
    attribute_query = sparql_attribute_extract % (entity, attribute)
    values = execute_sparql(attribute_query)
    return [
        v["value"]["value"]
        for v in values
        if "value" in v and isinstance(v["value"], dict) and "value" in v["value"]
    ] if values else []


# ---------------------------------------------------------------------------
# Entity name resolution
# ---------------------------------------------------------------------------


def get_entity_name(entity_id):
    """Get the English name of a Freebase entity, or ``None``."""
    sparql_query = f"""
    SELECT ?name WHERE {{
        <http://rdf.freebase.com/ns/{entity_id}> <http://rdf.freebase.com/ns/type.object.name> ?name .
        FILTER (lang(?name) = 'en')
    }}
    """
    result = execute_sparql(sparql_query)
    if result and isinstance(result, list) and len(result) > 0:
        name_dict = result[0].get("name", {})
        if isinstance(name_dict, dict) and "value" in name_dict:
            return name_dict["value"]
    return None


def get_entity_alias(entity_id):
    """Get the first English alias of a Freebase entity, or ``None``."""
    sparql_query = f"""
    SELECT ?alias WHERE {{
        <http://rdf.freebase.com/ns/{entity_id}> <http://rdf.freebase.com/ns/common.topic.alias> ?alias .
        FILTER (lang(?alias) = 'en')
    }}
    """
    result = execute_sparql(sparql_query)
    if result and isinstance(result, list) and len(result) > 0:
        for item in result:
            alias_dict = item.get("alias", {})
            if isinstance(alias_dict, dict) and "value" in alias_dict:
                return alias_dict["value"]
    return None


def get_entity_display_name(entity_id):
    """Return the best human-readable label for *entity_id*.

    Tries ``get_entity_name`` first, then ``get_entity_alias``, and falls
    back to the raw ID.
    """
    if not isinstance(entity_id, str) or not (
        entity_id.startswith("m.") or entity_id.startswith("g.")
    ):
        return str(entity_id)
    name = get_entity_name(entity_id)
    if name:
        return name
    alias = get_entity_alias(entity_id)
    if alias:
        return alias
    return entity_id


def id2entity_name_or_type(entity_id):
    """Map an entity ID to its name/type, returning ``'UnName_Entity'`` on failure."""
    sparql_query = sparql_id % (entity_id, entity_id)
    sparql = SPARQLWrapper(SPARQLPATH)
    sparql.setQuery(sparql_query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    if len(results["results"]["bindings"]) == 0:
        return "UnName_Entity"
    return results["results"]["bindings"][0]["tailEntity"]["value"]


# ---------------------------------------------------------------------------
# Relation filtering helpers
# ---------------------------------------------------------------------------


def abandon_rels(relation):
    """Return ``True`` if *relation* should be filtered out (meta / noise)."""
    if (
        relation == "type.object.type"
        or relation == "type.object.name"
        or relation == "type.object.key"
        or relation.startswith("freebase.")
        or "sameAs" in relation
        or "/" in relation
        or "topic.article" in relation
        or "topic.description" in relation
    ):
        return True
    return False


def check_end_word(s):
    """Check whether *s* ends with a known non-entity suffix."""
    words = [
        " ID", " code", " number", "instance of", "website",
        "URL", "inception", "image", " rate", " count",
    ]
    return any(s.endswith(word) for word in words)


# ---------------------------------------------------------------------------
# CRP parsing & SPARQL reconstruction
# ---------------------------------------------------------------------------


def _is_number(s):
    """Return ``True`` if *s* can be interpreted as a number."""
    try:
        float(s.replace(" ", "").replace(",", ""))
        return True
    except ValueError:
        return False


def _process_relation(relation):
    """Normalise a relation string extracted from a CRP triple."""
    if " , " in relation:
        if _is_number(relation):
            return relation.replace(" , ", ".").replace(" ,", ".").replace(", ", ".")
        else:
            return relation.replace(" , ", ",").replace(",", ".").replace(" ", "_")
    return relation


def parse_structured_kbqa_string(input_str):
    """Parse a CRP (Compositional Reasoning Path) string into a structured dict.

    Returns a dict with keys: ``main_entity``, ``main_path``,
    ``main_path_filters``, ``entity_restrictions``, ``numeric_restrictions``,
    ``string_restrictions``, ``other_restrictions``, ``target_variable``.
    """
    sections = {
        "main_path": [],
        "main_path_filters": [],
        "entity_restrictions": [],
        "numeric_restrictions": [],
        "string_restrictions": [],
        "other_restrictions": [],
    }

    current_section = None
    current_restriction = None

    triplet_pattern = r"^(\?[\w]+|[mg]\.\w+)\s+-\[(.+?)\]->\s+(\?[\w]+|[mg]\.\w+)$"
    filter_pattern = r"^\s*FILTER\s*\(\s*(.+)\s*\)$"
    order_by_pattern = r"^\s*ORDER BY\s+(ASC|DESC)\s*\(\s*(.+)\s*\)$"
    limit_pattern = r"^\s*LIMIT\s+(\d+)"

    section_patterns = {
        "main_path": r"^(main\s*path:)$",
        "main_path_filters": r"^(main\s*path\s*filters:)$",
        "entity_restrictions": r"^(entity\s*restrictions:)$",
        "numeric_restrictions": r"^(numeric\s*restrictions:)$",
        "string_restrictions": r"^(string\s*restrictions:)$",
        "other_restrictions": r"^(other\s*restrictions:)$",
    }

    for line in input_str.strip().splitlines():
        line = line.strip()
        if not line:
            continue

        # Check for section header
        matched_section = None
        for section_key, pattern in section_patterns.items():
            if re.match(pattern, line, re.IGNORECASE):
                matched_section = section_key
                break

        if matched_section:
            current_section = matched_section
            current_restriction = None
            continue

        if not current_section:
            continue

        # Try to parse a triple
        match = re.match(triplet_pattern, line)
        if match:
            left_entity, relation, right_entity = match.groups()
            processed_relation = _process_relation(relation)
            triple = (left_entity, processed_relation, right_entity)
            if current_section in [
                "main_path_filters",
                "numeric_restrictions",
                "string_restrictions",
            ]:
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
        if filter_match and current_section in [
            "main_path_filters",
            "numeric_restrictions",
            "string_restrictions",
        ]:
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

    # Extract main entity (first m.* or g.* entity in main_path)
    main_entity = None
    for left_entity, _, right_entity in sections["main_path"]:
        for entity in [left_entity, right_entity]:
            if entity.startswith("m.") or entity.startswith("g."):
                main_entity = entity
                break
        if main_entity:
            break

    # Target variable defaults to the object of the last main_path triple
    target_variable = "?x"
    if sections["main_path"]:
        target_variable = sections["main_path"][-1][2]
    if not main_entity:
        main_entity = "m.03_r3"

    # Remove empty restrictions
    for section in ["main_path_filters", "numeric_restrictions", "string_restrictions"]:
        sections[section] = [
            r for r in sections[section] if r.get("path") or r.get("filter")
        ]

    return {
        "main_entity": main_entity,
        "main_path": sections["main_path"],
        "main_path_filters": sections["main_path_filters"],
        "entity_restrictions": sections["entity_restrictions"],
        "numeric_restrictions": sections["numeric_restrictions"],
        "string_restrictions": sections["string_restrictions"],
        "other_restrictions": sections["other_restrictions"],
        "target_variable": target_variable,
    }


def reconstruct_sparql(parsed_result):
    """Convert a parsed CRP dict back into SPARQL queries.

    Returns a dict with ``simple_query`` and ``optimized_query`` (the latter
    applies date-range optimisation for temporal constraints).
    """
    main_entity = "ns:" + parsed_result["main_entity"]
    main_path = parsed_result["main_path"]
    entity_restrictions = parsed_result.get("entity_restrictions", [])
    numeric_restrictions = parsed_result.get("numeric_restrictions", [])
    string_restrictions = parsed_result.get("string_restrictions", [])
    other_restrictions = parsed_result.get("other_restrictions", [])
    target_var = "?x"

    def format_triple(s, p, o):
        s_full = "ns:" + s if not s.startswith("?") else s
        if o.startswith("?"):
            o_full = o
        elif o.startswith('"') and o.endswith('"'):
            o_full = o
        else:
            o_full = "ns:" + o

        if p.startswith("R[") and p.endswith("]"):
            prop = p[2:-1]
            return f"  {o_full} ns:{prop} {s_full} ."
        else:
            p_full = "ns:" + p if not p.startswith("ns:") else p
            return f"  {s_full} {p_full} {o_full} ."

    def add_numeric_suffix(filter_str):
        """Add XSD type suffixes to numeric/date literals in FILTER clauses."""
        # Date format yyyy-mm-dd
        date_pattern = r'(\?[\w]+)\s(<=|>=|<|>|=)\s"(\d{4}-\d{2}-\d{2})"'

        def process_date_filter(match):
            var, op, date = match.group(1), match.group(2), match.group(3)
            if op == "=":
                return f'regex(str({var}), "^{date}")'
            return f'{var} {op} "{date}"^^xsd:dateTime'

        filter_str = re.sub(date_pattern, process_date_filter, filter_str)

        # Year (4-digit)
        year_pattern = r'(\?[\w]+)\s(<=|>=|<|>|=)\s"(\d{4})"(?!\.\d|\d|\^)'

        def process_year_filter(match):
            var, op, year = match.group(1), match.group(2), match.group(3)
            if op == "=":
                return f'regex(str({var}), "^{year}")'
            return f'{var} {op} "{year}"^^xsd:dateTime'

        filter_str = re.sub(year_pattern, process_year_filter, filter_str)

        # Explicit floats
        float_pattern = r'(\?[\w]+)\s(<=|>=|=|<|>)\s"(-?(?:\d+\.\d+|\.\d+))"(?!\^\^)'
        filter_str = re.sub(float_pattern, r'\1 \2 "\3"^^xsd:float', filter_str)

        # Generic numbers (integers, scientific notation)
        generic_pattern = r'(\?[\w]+)\s(<=|>=|<|>|=)\s"(-?[\d.eE+\-]+)"(?!\^\^)'

        def process_generic(match):
            var, op, number = match.group(1), match.group(2), match.group(3)
            if op == "=":
                return f'regex(str({var}), "^{number}")'
            return f'{var} {op} "{number}"^^xsd:float'

        filter_str = re.sub(generic_pattern, process_generic, filter_str)
        return filter_str

    # Check for ?c variable
    has_c_variable = False
    all_triples = main_path + entity_restrictions + other_restrictions
    for s, _, o in all_triples:
        if s == "?c" or o == "?c":
            has_c_variable = True
            break

    # Collect ORDER BY / LIMIT
    order_by_clause = None
    limit_clause = None
    for restriction in numeric_restrictions:
        if "order_by" in restriction:
            order_by_str = restriction["order_by"]
            order_by_str = re.sub(
                r"ORDER BY (DESC|ASC)\((\?\w+)\)",
                r"ORDER BY \1((\2))",
                order_by_str,
            )
            order_by_clause = order_by_str
        if "limit" in restriction:
            limit_clause = restriction["limit"]

    # --- Simple query ---
    sparql_simple = [
        "PREFIX ns: <http://rdf.freebase.com/ns/>",
        f"SELECT DISTINCT {target_var}",
        "WHERE {",
    ]
    for s, p, o in main_path:
        sparql_simple.append(format_triple(s, p, o))
    for s, p, o in entity_restrictions:
        sparql_simple.append(format_triple(s, p, o))
    for restriction in numeric_restrictions:
        for s, p, o in restriction.get("path", []):
            sparql_simple.append(format_triple(s, p, o))
        if "filter" in restriction:
            sparql_simple.append(f"  {add_numeric_suffix(restriction['filter'])}")
    for restriction in string_restrictions:
        for s, p, o in restriction.get("path", []):
            sparql_simple.append(format_triple(s, p, o))
        if "filter" in restriction:
            sparql_simple.append(f"  {restriction['filter']}")
    for s, p, o in other_restrictions:
        sparql_simple.append(format_triple(s, p, o))

    sparql_simple.append(
        f"  FILTER ({target_var} != {'?c' if has_c_variable else main_entity})"
    )
    sparql_simple.append(
        f'  FILTER (!isLiteral({target_var}) || lang({target_var}) = "" '
        f'|| langMatches(lang({target_var}), "en"))'
    )
    sparql_simple.append("}")
    if order_by_clause:
        sparql_simple.append(order_by_clause)
    if limit_clause:
        sparql_simple.append(limit_clause)

    # --- Optimised query (date-range awareness) ---
    sparql_opt = [
        "PREFIX ns: <http://rdf.freebase.com/ns/>",
        f"SELECT DISTINCT {target_var}",
        "WHERE {",
    ]
    for s, p, o in main_path:
        sparql_opt.append(format_triple(s, p, o))
    for s, p, o in entity_restrictions:
        sparql_opt.append(format_triple(s, p, o))
    for restriction in string_restrictions:
        for s, p, o in restriction.get("path", []):
            sparql_opt.append(format_triple(s, p, o))
        if "filter" in restriction:
            sparql_opt.append(f"  {restriction['filter']}")
    for s, p, o in other_restrictions:
        sparql_opt.append(format_triple(s, p, o))

    # Separate date vs other numeric restrictions
    has_date_filter = False
    date_restrictions = []
    non_date_restrictions = []
    from_var, to_var, from_prop, to_prop = None, None, None, None
    start_date, end_date = None, None

    for restriction in numeric_restrictions:
        filter_str = restriction.get("filter", "")
        path = restriction.get("path", [])
        is_date = False
        if len(path) > 0:
            prop = path[0][1]
            if re.search(r"(from|start_date)", prop):
                dm = re.search(r'"(\d{4}-\d{2}-\d{2})"', filter_str)
                if dm:
                    is_date = True
                    has_date_filter = True
                    from_var = path[0][2]
                    from_prop = prop
                    start_date = dm.group(1)
            elif re.search(r"(to|end_date)", prop):
                dm = re.search(r'"(\d{4}-\d{2}-\d{2})"', filter_str)
                if dm:
                    is_date = True
                    has_date_filter = True
                    to_var = path[0][2]
                    to_prop = prop
                    end_date = dm.group(1)
        if is_date:
            date_restrictions.append(restriction)
        else:
            non_date_restrictions.append(restriction)

    if has_date_filter:
        var = "?y"
        year = start_date[:4] if start_date else end_date[:4]
        ref_start = start_date if start_date else end_date
        ref_end = end_date if end_date else start_date
        sparql_opt.extend([
            f"  FILTER (",
            f'    (!BOUND(?from) || xsd:dateTime(?from) <= "{ref_start}"^^xsd:dateTime) &&',
            f'    (!BOUND(?to) || xsd:dateTime(?to) >= "{ref_end}"^^xsd:dateTime)',
            f"  )",
            f"  OPTIONAL {{ {var} ns:{from_prop} ?from . }}",
            f"  OPTIONAL {{ {var} ns:{to_prop} ?to . }}",
        ])

    for restriction in non_date_restrictions:
        for s, p, o in restriction.get("path", []):
            sparql_opt.append(format_triple(s, p, o))
        if "filter" in restriction:
            sparql_opt.append(f"  {add_numeric_suffix(restriction['filter'])}")

    sparql_opt.append(
        f"  FILTER ({target_var} != {'?c' if has_c_variable else main_entity})"
    )
    sparql_opt.append(
        f'  FILTER (!isLiteral({target_var}) || lang({target_var}) = "" '
        f'|| langMatches(lang({target_var}), "en"))'
    )
    sparql_opt.append("}")
    if order_by_clause:
        sparql_opt.append(order_by_clause)
    if limit_clause:
        sparql_opt.append(limit_clause)

    return {
        "simple_query": "\n".join(sparql_simple),
        "optimized_query": "\n".join(sparql_opt),
    }


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------


def get_embedding(texts, batch_size=70):
    """Compute embeddings using a local Ollama server (nomic-embed-text).

    Args:
        texts: A single string or list of strings to embed.
        batch_size: Number of texts per batch.

    Returns:
        A numpy array of embeddings, or ``None`` on failure.
    """
    import ollama

    ollama_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    ollama_client = ollama.Client(host=ollama_host)
    embed_model = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text:latest")

    if isinstance(texts, str):
        try:
            response = ollama_client.embeddings(model=embed_model, prompt=texts)
            return np.array(response["embedding"])
        except Exception as e:
            print(f"Ollama embedding error: {e}")
            return None

    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        for text in batch_texts:
            try:
                response = ollama_client.embeddings(model=embed_model, prompt=text)
                embeddings.append(response["embedding"])
            except Exception as e:
                print(f"Ollama embedding error for '{text}': {e}")
                embeddings.append(None)

    if all(emb is None for emb in embeddings):
        return None
    return np.array(embeddings)


# ---------------------------------------------------------------------------
# LLM-based path selection
# ---------------------------------------------------------------------------


def _build_relation_path(relations):
    """Join relation list into a dotted path string (short form)."""
    if not relations:
        return ""
    parts = relations[0].split(".")
    path = ".".join(parts[-2:])
    for rel in relations[1:]:
        parts = rel.split(".")
        path += " -> " + ".".join(parts[-2:])
    return path


def select_paths_with_llm(
    question, top_m_relations, initial_entity, triplets,
    llm_type, temperature, max_tokens, api_key,
    n=3, is_final_step=False,
):
    """Use an LLM to select the most promising paths from the candidate set.

    Uses the unified path selection prompt from the paper (Appendix A.3).
    At intermediate beam search steps, *n* is set to the beam width *w*;
    at the final step (maximum depth reached), *n* = 1.

    Args:
        question: The natural-language question.
        top_m_relations: Candidate relation paths (list of dicts with ``'path'``).
        initial_entity: Starting entity ID(s).
        triplets: Already-collected knowledge triples (unused, kept for API compat).
        llm_type: Model identifier for ``run_llm``.
        temperature: Sampling temperature.
        max_tokens: Maximum generation tokens.
        api_key: API key for the LLM service.
        n: Number of paths to select (beam width *w* for intermediate, 1 for final).
        is_final_step: If True, override *n* to 1.

    Returns:
        List of selected relation path lists.
    """
    from .llm_utils import run_llm

    if is_final_step:
        n = 1

    entity_name_cache = {}
    for entity_id in initial_entity:
        entity_name_cache[entity_id] = get_entity_display_name(entity_id) or entity_id

    start_entity_names_str = ", ".join(entity_name_cache.values())

    # Build path descriptions
    path_descriptions = []
    for i, relation_path in enumerate(top_m_relations):
        path_str = _build_relation_path(relation_path["path"])
        path_first_relation = relation_path["path"][0]

        start_entities = []
        for entity_id in initial_entity:
            if entity_search(entity_id, path_first_relation, head=True):
                start_entities.append(entity_name_cache[entity_id])
                continue
            if entity_search(entity_id, path_first_relation, head=False):
                start_entities.append(entity_name_cache[entity_id])
                continue
            if attribute_search(entity_id, path_first_relation):
                start_entities.append(entity_name_cache[entity_id])
                continue

        start_str = ", ".join(start_entities) if start_entities else start_entity_names_str

        end_entities = relation_path.get("end_entity", [])
        end_entity_names = [
            entity_name_cache.get(eid) or get_entity_display_name(eid) or eid
            for eid in end_entities
            if isinstance(eid, str)
        ]
        end_str = ", ".join(end_entity_names) if end_entity_names else "unknown entity"

        path_desc = f"{start_str} -> {path_str} -> {end_str}..."
        path_descriptions.append(f"Path {i + 1}: {path_desc}")

    paths_text = "\n".join(path_descriptions)
    if not paths_text:
        print("No available paths, skipping LLM selection")
        return [p["path"] for p in top_m_relations[:n]]

    # Unified prompt from paper (Appendix A.3)
    prompt = (
        f"Q: '{question}'\n"
        f"Paths from '{start_entity_names_str}':\n"
        f"{paths_text}\n\n"
        f"Select up to {n} paths most likely to reach the FINAL answer "
        f"(not intermediate entities). Reply with path numbers only, "
        f"e.g.: Path 1, Path 2"
    )

    response = run_llm(prompt, temperature, max_tokens, api_key, llm_type)

    selected_relations = []
    for i in range(len(top_m_relations)):
        pattern = rf"\bPath {i + 1}\b"
        if re.search(pattern, response):
            selected_relations.append(top_m_relations[i]["path"])
            if len(selected_relations) >= n:
                break

    if not selected_relations:
        print("LLM did not select any paths, returning top paths by similarity")
        return [p["path"] for p in top_m_relations[:n]]

    print(f"Selected {len(selected_relations)} path(s): {selected_relations}")
    return selected_relations


def change_paths_with_llm_cwq(
    question, existing_paths, main_path, initial_entity,
    selected_path, llm_type, temperature, max_tokens, api_key,
):
    """Final-step path selection: choose 1 best path from main + candidates.

    Uses the same unified prompt template as ``select_paths_with_llm``
    (paper Appendix A.3) with n=1. The main path is included as Path 1,
    and candidates follow as Path 2, 3, etc.

    Returns:
        ``(selected_path_dict, should_replace)`` -- the chosen path and whether
        it differs from the main path.
    """
    from .llm_utils import run_llm

    if isinstance(initial_entity, str):
        initial_entity = [initial_entity]

    entity_name_cache = {}
    for entity_id in initial_entity:
        entity_name_cache[entity_id] = get_entity_display_name(entity_id) or entity_id

    start_entity_names_str = ", ".join(entity_name_cache.values())

    # Build combined path list: main path first, then candidates
    all_paths = [main_path] + [
        p for p in existing_paths
        if _build_relation_path(p.get("path", [])) != _build_relation_path(main_path.get("path", []))
    ]

    path_descriptions = []
    for i, path_dict in enumerate(all_paths):
        path_str = _build_relation_path(path_dict.get("path", []))
        end_entities = path_dict.get("end_entity", [])
        end_names = []
        for eid in end_entities:
            if isinstance(eid, str):
                if eid not in entity_name_cache:
                    entity_name_cache[eid] = get_entity_display_name(eid) or eid
                end_names.append(entity_name_cache[eid])
        end_str = ", ".join(end_names) if end_names else "unknown entity"
        path_descriptions.append(
            f"Path {i + 1}: {start_entity_names_str} -> {path_str} -> {end_str}..."
        )

    paths_text = "\n".join(path_descriptions)
    if not paths_text or len(all_paths) <= 1:
        return main_path, False

    # Unified prompt (n=1, final step)
    prompt = (
        f"Q: '{question}'\n"
        f"Paths from '{start_entity_names_str}':\n"
        f"{paths_text}\n\n"
        f"Select up to 1 paths most likely to reach the FINAL answer "
        f"(not intermediate entities). Reply with path numbers only, "
        f"e.g.: Path 1"
    )

    response = run_llm(prompt, temperature, max_tokens, api_key, llm_type)

    # Parse selected path number
    for i in range(len(all_paths)):
        pattern = rf"\bPath {i + 1}\b"
        if re.search(pattern, response):
            result_path = all_paths[i]
            should_replace = (i != 0)  # True if not main path
            if should_replace:
                print(f"Final selection: replacing main path with Path {i + 1}")
            else:
                print("Final selection: keeping main path (Path 1)")
            return result_path, should_replace

    print("No valid path selected in final step, keeping main path")
    return main_path, False

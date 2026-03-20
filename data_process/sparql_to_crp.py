"""
SPARQL to CRP (Constraint Reasoning Path) converter.

Parses SPARQL queries into a structured representation consisting of:
- A main entity and main path (chain of triples from entity to target variable)
- Entity restrictions (triples constraining to specific entities)
- Numeric restrictions (filters on numeric/date values, ORDER BY, LIMIT)
- String restrictions (filters on string literal values)
- Other restrictions (remaining unclassified triples)
"""

import re
from collections import defaultdict, deque


def parse_all_filters(sparql_query, triples, main_path_vars):
    """Extract and classify all FILTER clauses from a SPARQL query.

    Finds FILTER expressions in the query, determines which variables they
    reference, traces paths from main-path variables to those filter variables
    through the triple graph, and returns structured restriction objects.

    Args:
        sparql_query: The full SPARQL query string.
        triples: List of (subject, predicate, object) tuples extracted from
            the WHERE clause.
        main_path_vars: Set of variable names that appear on the main path.

    Returns:
        A list of restriction dicts, each with keys:
            - "type": "string" or "entity" (optional for numeric)
            - "path": list of (s, p, o) tuples with 'ns:' prefixes stripped
            - "filter": the FILTER expression string
    """
    restrictions = []
    processed_filters = set()

    all_possible_triples = triples.copy()

    # Extract additional triples from EXISTS clauses
    exists_pattern = r'EXISTS\s*\{([^}]+)\}'
    exists_matches = re.findall(exists_pattern, sparql_query, re.DOTALL)
    for exists_content in exists_matches:
        triple_pattern = (
            r'(\?\w+|ns:[^\s]+)\s+(ns:[^\s]+)\s+'
            r'(\?\w+|ns:[^\s]+|"[^"]*"\^\^xsd:dateTime|"[^"]*")\s*\.'
        )
        exists_triples = re.findall(triple_pattern, exists_content)
        for s, p, o in exists_triples:
            if (s, p, o) not in all_possible_triples:
                all_possible_triples.append((s, p, o))

    # Build variable adjacency graph
    var_graph = defaultdict(list)
    triple_map = defaultdict(list)
    for s, p, o in all_possible_triples:
        if s.startswith('?') and o.startswith('?'):
            var_graph[s].append(o)
            var_graph[o].append(s)
            triple_map[(s, o)].append((s, p, o))
            triple_map[(o, s)].append((s, p, o))

    def find_path_to_main_path(start_var, max_hops=2):
        """BFS from main-path variables to start_var, returning the triple path."""
        visited = set()
        queue = deque([(v, []) for v in main_path_vars])
        while queue:
            current_var, path = queue.popleft()
            if current_var in visited:
                continue
            visited.add(current_var)
            if current_var == start_var and path:
                return path
            if len(path) >= max_hops * 3:
                continue
            for neighbor in var_graph.get(current_var, []):
                for triple in triple_map[(current_var, neighbor)]:
                    if triple not in [t for p in path for t in p]:
                        queue.append((neighbor, path + [triple]))
        return None

    # Scan for FILTER clauses by balanced parenthesis matching
    pos = 0
    while True:
        filter_start = sparql_query.find('FILTER', pos)
        if filter_start == -1:
            break
        paren_start = sparql_query.find('(', filter_start)
        if paren_start == -1:
            pos = filter_start + 6
            continue
        paren_count = 1
        current_pos = paren_start + 1
        while current_pos < len(sparql_query) and paren_count > 0:
            char = sparql_query[current_pos]
            if char == '(':
                paren_count += 1
            elif char == ')':
                paren_count -= 1
            current_pos += 1

        if paren_count == 0:
            filter_content = sparql_query[paren_start + 1:current_pos - 1]

            # Check for string filter: str(?var) = "value"
            string_filter_match = re.match(
                r'str\s*\(\s*(\?\w+)\s*\)\s*=\s*"([^"]*)"',
                filter_content.strip()
            )
            # Check for entity filter: ?var != ns:entity or ?var = ns:entity
            entity_filter_match = re.match(
                r'\s*(\?\w+)\s*(!=|=)\s*(ns:[^\s]+)',
                filter_content.strip()
            )

            if string_filter_match:
                var_name = string_filter_match.group(1)
                string_value = string_filter_match.group(2)
                content_key = filter_content.strip()
                if content_key in processed_filters:
                    pos = current_pos
                    continue
                processed_filters.add(content_key)

                path = find_path_to_main_path(var_name)
                if path:
                    restriction = {
                        "type": "string",
                        "path": [(s.replace('ns:', ''), p.replace('ns:', ''), o)
                                 for s, p, o in path],
                        "filter": f'FILTER (str({var_name}) = "{string_value}")'
                    }
                    restrictions.append(restriction)
                pos = current_pos
                continue

            elif entity_filter_match:
                var_name = entity_filter_match.group(1)
                operator = entity_filter_match.group(2)
                entity_value = entity_filter_match.group(3)
                content_key = filter_content.strip()
                if content_key in processed_filters:
                    pos = current_pos
                    continue
                processed_filters.add(content_key)

                path = find_path_to_main_path(var_name)
                if path:
                    restriction = {
                        "type": "entity",
                        "path": [(s.replace('ns:', ''), p.replace('ns:', ''), o)
                                 for s, p, o in path],
                        "filter": f"FILTER ({var_name} {operator} {entity_value})"
                    }
                    restrictions.append(restriction)
                pos = current_pos
                continue

            # Handle nested or numeric filters
            nested_filters = []
            inner_pos = 0
            while True:
                inner_filter_start = filter_content.find('FILTER', inner_pos)
                if inner_filter_start == -1:
                    break
                inner_paren_start = filter_content.find('(', inner_filter_start)
                if inner_paren_start == -1:
                    inner_pos = inner_filter_start + 6
                    continue
                inner_paren_count = 1
                inner_current_pos = inner_paren_start + 1
                while inner_current_pos < len(filter_content) and inner_paren_count > 0:
                    char = filter_content[inner_current_pos]
                    if char == '(':
                        inner_paren_count += 1
                    elif char == ')':
                        inner_paren_count -= 1
                    inner_current_pos += 1
                if inner_paren_count == 0:
                    nested_filter_content = filter_content[
                        inner_paren_start + 1:inner_current_pos - 1
                    ]
                    nested_filters.append(nested_filter_content)
                inner_pos = inner_current_pos

            if not nested_filters:
                if (re.search(r'"[^"]*\d+[^"]*"', filter_content)
                        or re.search(r'\d+', filter_content)):
                    nested_filters.append(filter_content)

            # Merge paths for filters referencing the same variables
            filter_paths = {}
            for content in nested_filters:
                content_key = content.strip()
                if content_key in processed_filters:
                    continue
                processed_filters.add(content_key)
                variables = re.findall(r'\?(\w+)', content)

                for var in variables:
                    var_name = f"?{var}"
                    path = find_path_to_main_path(var_name)
                    if path:
                        simplified_filter = re.sub(
                            r'xsd:datetime\s*\(\s*(\?\w+)\s*\)', r'\1', content
                        )
                        simplified_filter = re.sub(
                            r'\^\^xsd:dateTime', '', simplified_filter
                        )
                        filter_paths[simplified_filter] = (
                            filter_paths.get(simplified_filter, []) + path
                        )

            # Sort paths from main-path variables outward
            for simplified_filter, path in filter_paths.items():
                sorted_path = []
                current_vars = set(main_path_vars)
                while path:
                    found = False
                    for triple in path:
                        s, p, o = triple
                        if s in current_vars:
                            sorted_path.append(triple)
                            path.remove(triple)
                            current_vars.add(o)
                            found = True
                            break
                        elif o in current_vars:
                            sorted_path.append(triple)
                            path.remove(triple)
                            current_vars.add(s)
                            found = True
                            break
                    if not found:
                        break
                if sorted_path:
                    restriction = {
                        "path": [(s.replace('ns:', ''), p.replace('ns:', ''), o)
                                 for s, p, o in sorted_path],
                        "filter": f"FILTER ({simplified_filter})"
                    }
                    restrictions.append(restriction)

        pos = current_pos

    return restrictions


def _is_datetime_literal(value):
    """Check whether value is an xsd:dateTime literal."""
    return bool(re.match(r'"(\d{4}(?:-\d{2}){0,2})"\^\^xsd:dateTime', value))


def _is_string_literal(value):
    """Check whether value is a string literal (possibly with a language tag)."""
    if re.match(r'"[^"]*"@\w+', value):
        return True
    if re.match(r'"[^"]*"$', value) and not re.match(
        r'"[0-9.-]+(?:[eE][+-]?[0-9]+)?"$', value
    ):
        return True
    return False


def _is_numeric_literal(value):
    """Check whether value is a numeric or dateTime literal."""
    if _is_datetime_literal(value):
        return True
    if re.match(r'".*"\^\^xsd:float', value):
        return True
    if re.match(r'"[0-9.-]+(?:[eE][+-]?[0-9]+)?"$', value):
        return True
    return False


def _extract_numeric_value(value):
    """Extract the raw numeric or date string from a literal."""
    if _is_datetime_literal(value):
        match = re.match(
            r'"(\d{4}-\d{2}-\d{2}|\d{4}-\d{2}|\d{4})"\^\^xsd:dateTime', value
        )
        if match:
            return match.group(1)
    elif re.match(r'".*"\^\^xsd:float', value):
        match = re.match(r'"([^"]*)"\^\^xsd:float', value)
        if match:
            return match.group(1)
    elif re.match(r'"[0-9.-]+(?:[eE][+-]?[0-9]+)?"$', value):
        return value.strip('"')
    return None


def _extract_string_value(value):
    """Extract the raw string from a string literal."""
    match = re.match(r'"([^"]*)"@\w+', value)
    if match:
        return match.group(1)
    match = re.match(r'"([^"]*)"$', value)
    if match:
        return match.group(1)
    return None


def parse_sparql(sparql_query):
    """Parse a SPARQL query into a Constraint Reasoning Path (CRP) representation.

    Extracts the structural components of a Freebase SPARQL query and returns
    a dictionary describing the main entity, the chain of relations leading to
    the target variable, and any constraint branches (entity, numeric, string).

    The parser handles:
    - UNION clauses (keeps only the left branch)
    - FILTER expressions (numeric comparisons, string matches, entity filters)
    - ORDER BY / LIMIT modifiers
    - Reverse relations (encoded as ``R[relation]``)

    Args:
        sparql_query: A SPARQL query string using ``ns:`` prefixed Freebase
            identifiers (e.g. ``ns:m.0abc``).

    Returns:
        A dict with the following keys, or ``None`` if parsing fails:

        - ``main_entity`` (str): The Freebase ID of the topic entity
          (without ``ns:`` prefix).
        - ``main_path`` (list[tuple]): Ordered list of ``(s, p, o)`` triples
          from the main entity to the target variable. Reverse edges use
          ``R[predicate]`` notation for the predicate.
        - ``main_path_filters`` (list[dict]): Filters that apply directly to
          the target variable (e.g. inequality exclusions).
        - ``entity_restrictions`` (list[tuple]): ``(s, p, o)`` triples forming
          constraint branches that end at a specific entity.
        - ``numeric_restrictions`` (list[dict]): Dicts with ``"path"``,
          optional ``"filter"``, ``"order_by"``, ``"limit"`` keys.
        - ``string_restrictions`` (list[dict]): Dicts with ``"path"`` and
          ``"filter"`` keys for string-valued constraints.
        - ``other_restrictions`` (list[tuple]): Any remaining unclassified
          triples.
        - ``target_variable`` (str): The SELECT variable (e.g. ``?x``).
    """
    triples = []
    main_entity = None
    main_path = []
    entity_restrictions = []
    numeric_restrictions = []
    string_restrictions = []
    other_restrictions = []
    main_path_filters = []

    # Step 1: Extract the WHERE clause using balanced-brace matching
    where_start_match = re.search(
        r'WHERE\s*\{', sparql_query, re.IGNORECASE | re.DOTALL
    )
    if not where_start_match:
        return None
    start = where_start_match.end() - 1
    brace_count = 1
    i = start + 1
    while i < len(sparql_query) and brace_count > 0:
        if sparql_query[i] == '{':
            brace_count += 1
        elif sparql_query[i] == '}':
            brace_count -= 1
        i += 1
    if brace_count != 0:
        return None
    where_clause = sparql_query[start:i]

    # Step 2: Remove the right-hand side of UNION blocks
    def process_where_clause(content):
        new_content = ''
        j = 0
        n = len(content)
        while j < n:
            if re.match(r'\bUNION\b', content[j:j + 5]):
                j += 5
                while j < n and content[j].isspace():
                    j += 1
                if j < n and content[j] == '{':
                    j += 1
                    inner_brace_count = 1
                    while j < n and inner_brace_count > 0:
                        if content[j] == '{':
                            inner_brace_count += 1
                        elif content[j] == '}':
                            inner_brace_count -= 1
                        j += 1
            else:
                new_content += content[j]
                j += 1
        return new_content

    inner_content = where_clause[1:-1].strip()
    processed_content = process_where_clause(inner_content)

    # Extract ORDER BY and LIMIT information
    order_by_match = re.search(
        r'ORDER BY\s+(DESC|ASC)?\s*\(?(?:xsd:\w+\()?([^)\s]+)(?:\))?\)?',
        sparql_query,
    )
    limit_match = re.search(r'LIMIT\s+(\d+)', sparql_query)

    # Collect non-comment, non-empty lines
    lines = []
    for line in processed_content.split('\n'):
        clean_line = line.strip()
        if clean_line and not clean_line.startswith('#'):
            clean_line = re.sub(r'; #.*$', '.', clean_line)
            lines.append(clean_line)

    full_content = ' '.join(lines)

    # Extract triples (supports language tags like @en)
    triple_pattern = (
        r'(\?\w+|ns:[^\s]+)\s+(ns:[^\s]+)\s+'
        r'(\?\w+|ns:[^\s]+|<[^>]+>|"[^"]*"(?:\^\^xsd:\w+|@\w+)?|"[^"]*")\s*\.'
    )
    triple_matches = re.findall(triple_pattern, full_content)
    basic_triples = [(s, p, o) for s, p, o in triple_matches]

    triples = basic_triples

    # Identify the main (topic) entity
    for s, p, o in triples:
        if s.startswith('ns:m.') or s.startswith('ns:g.'):
            main_entity = s
            break
        elif o.startswith('ns:m.') or o.startswith('ns:g.'):
            main_entity = o
            break
    if not main_entity:
        return None

    # Determine the target SELECT variable
    visited_triples = set()
    current = main_entity
    select_match = re.search(r'SELECT\s+DISTINCT\s+(\?\w+)', sparql_query)
    target_var = select_match.group(1) if select_match else "?x"

    # Step 3: Trace the main path from the topic entity to the target variable
    while True:
        found_next = False
        for idx, (s, p, o) in enumerate(triples):
            triple_tuple = (s, p, o)
            if triple_tuple in visited_triples:
                continue
            if s == current and o.startswith('?'):
                main_path.append((s, p, o))
                visited_triples.add(triple_tuple)
                current = o
                found_next = True
                if current == target_var:
                    found_next = False
                break
            elif o == current and s.startswith('?'):
                main_path.append((current, 'R[' + p + ']', s))
                visited_triples.add(triple_tuple)
                current = s
                found_next = True
                if current == target_var:
                    found_next = False
                break
        if not found_next:
            break

    # Collect variables on the main path
    main_path_vars = set()
    for s, p, o in main_path:
        if s.startswith('?'):
            main_path_vars.add(s)
        if o.startswith('?'):
            main_path_vars.add(o)

    filter_restrictions = parse_all_filters(sparql_query, basic_triples, main_path_vars)

    # Process ORDER BY related variables and triples
    order_by_info = None
    limit_info = None
    order_var = None
    order_by_triples = set()

    if order_by_match:
        order_dir = order_by_match.group(1) or "ASC"
        order_var = order_by_match.group(2)
        if order_var:
            order_by_info = f"ORDER BY {order_dir}({order_var})"
            for triple in triples:
                s, p, o = triple
                if s == order_var or o == order_var:
                    order_by_triples.add(triple)

    if limit_match:
        limit_info = f"LIMIT {limit_match.group(1)}"

    path_variables = {target_var}
    for s, p, o in main_path:
        if s.startswith('?'):
            path_variables.add(s)
        if o.startswith('?'):
            path_variables.add(o)

    # Build a variable graph from remaining (non-main-path) triples
    var_graph = defaultdict(list)
    triple_map = defaultdict(list)

    for triple in triples:
        if triple in visited_triples:
            continue
        s, p, o = triple
        if s.startswith('?'):
            var_graph[s].append(o)
            triple_map[(s, o)].append(triple)
        if o.startswith('?'):
            var_graph[o].append(s)
            triple_map[(o, s)].append(triple)

    # BFS from main-path variables to find entity restriction branches
    visited_vars = set()
    used_triples = set()

    for start_var in path_variables:
        queue = deque([(start_var, [])])
        while queue:
            current_var, path = queue.popleft()
            if current_var in visited_vars:
                continue
            visited_vars.add(current_var)
            for neighbor in var_graph.get(current_var, []):
                if neighbor.startswith('ns:m.') or neighbor.startswith('ns:g.'):
                    for t in path:
                        entity_restrictions.append(t)
                        used_triples.add(t)
                    for triple in triple_map[(current_var, neighbor)]:
                        entity_restrictions.append(triple)
                        used_triples.add(triple)
                    continue
                elif neighbor.startswith('?') and len(path) < 1:
                    for triple in triple_map[(current_var, neighbor)]:
                        queue.append((neighbor, path + [triple]))

    remaining_triples = [
        t for t in triples if t not in visited_triples and t not in used_triples
    ]
    remaining_used = set()

    # Handle main-path-level filters (e.g. target variable exclusions)
    for restriction in filter_restrictions:
        if (restriction.get("type") == "entity"
                and restriction.get("filter", "").startswith(
                    f"FILTER ({target_var} != ")):
            main_path_filters.append(restriction)
            for triple_info in restriction.get("path", []):
                for triple in remaining_triples:
                    s, p, o = triple
                    if (s.replace('ns:', '') == triple_info[0]
                            and p.replace('ns:', '') == triple_info[1]
                            and o == triple_info[2]):
                        remaining_used.add(triple)
            filter_restrictions.remove(restriction)

    # Handle ORDER BY: find path from main-path variables to the order variable
    if order_by_info and order_var:
        visited = set()
        queue = deque([(v, []) for v in path_variables])
        order_path_triples = []
        found = False
        while queue:
            current_node, path = queue.popleft()
            if current_node in visited:
                continue
            visited.add(current_node)
            if current_node == order_var:
                order_path_triples = path
                found = True
                break
            for neighbor in var_graph.get(current_node, []):
                for triple in triple_map[(current_node, neighbor)]:
                    if triple not in [t for p in path for t in [p]]:
                        queue.append((neighbor, path + [triple]))

        if found:
            order_by_path = [
                (s.replace('ns:', ''), p.replace('ns:', ''), o)
                for s, p, o in order_path_triples
            ]
            for t in order_path_triples:
                remaining_used.add(t)
            restriction = {"path": order_by_path, "order_by": order_by_info}
            if limit_info:
                restriction["limit"] = limit_info
            numeric_restrictions.append(restriction)

    # Rebuild main_path_vars for constraint classification
    main_path_vars = set()
    for s, p, o in main_path:
        if s.startswith('?'):
            main_path_vars.add(s)
        if o.startswith('?'):
            main_path_vars.add(o)

    # Classify remaining triples into string or numeric restrictions
    processed_pairs = set()

    # -- String restrictions (one-step and two-step) --
    for (s1, p1, o1) in remaining_triples:
        if (s1, p1, o1) in remaining_used:
            continue

        # One-step: variable -> string literal
        if s1.startswith('?') and _is_string_literal(o1):
            string_value = _extract_string_value(o1)
            if string_value:
                name_var = "?name"
                string_restrictions.append({
                    "path": [(s1.replace('ns:', ''), p1.replace('ns:', ''), name_var)],
                    "filter": f'FILTER ({name_var} = "{string_value}")'
                })
                remaining_used.add((s1, p1, o1))
                continue

        # Two-step: variable -> intermediate variable -> string literal
        if not o1.startswith('?'):
            continue

        for (s2, p2, o2) in remaining_triples:
            if (s2, p2, o2) in remaining_used:
                continue
            if s2 == o1 and _is_string_literal(o2):
                pair_key = (s1, p1, o1, s2, p2, o2)
                if pair_key in processed_pairs:
                    continue
                string_value = _extract_string_value(o2)
                if string_value:
                    name_var = "?name"
                    path = [(s1, p1, o1), (s2, p2, name_var)]
                    string_restrictions.append({
                        "path": [
                            (s.replace('ns:', ''), p.replace('ns:', ''),
                             o if o.startswith('?') else name_var)
                            for s, p, o in path
                        ],
                        "filter": f'FILTER ({name_var} = "{string_value}")'
                    })
                    remaining_used.add((s1, p1, o1))
                    remaining_used.add((s2, p2, o2))
                    processed_pairs.add(pair_key)
                    break

    # -- Numeric restrictions (one-step and two-step) --
    for (s1, p1, o1) in remaining_triples:
        if (s1, p1, o1) in remaining_used:
            continue

        # One-step: variable -> numeric literal
        if s1.startswith('?') and _is_numeric_literal(o1):
            numeric_value = _extract_numeric_value(o1)
            if numeric_value:
                number_var = "?number"
                numeric_restrictions.append({
                    "path": [(s1.replace('ns:', ''), p1.replace('ns:', ''), number_var)],
                    "filter": f'FILTER ({number_var} = "{numeric_value}")'
                })
                remaining_used.add((s1, p1, o1))
                continue

        # Two-step: variable -> intermediate variable -> numeric literal
        if not o1.startswith('?'):
            continue

        for (s2, p2, o2) in remaining_triples:
            if (s2, p2, o2) in remaining_used:
                continue
            if s2 == o1 and _is_numeric_literal(o2):
                pair_key = (s1, p1, o1, s2, p2, o2)
                if pair_key in processed_pairs:
                    continue
                numeric_value = _extract_numeric_value(o2)
                if numeric_value:
                    number_var = "?number"
                    path = [(s1, p1, o1), (s2, p2, number_var)]
                    numeric_restrictions.append({
                        "path": [
                            (s.replace('ns:', ''), p.replace('ns:', ''),
                             o if o.startswith('?') else number_var)
                            for s, p, o in path
                        ],
                        "filter": f'FILTER ({number_var} = "{numeric_value}")'
                    })
                    remaining_used.add((s1, p1, o1))
                    remaining_used.add((s2, p2, o2))
                    processed_pairs.add(pair_key)
                    break

    # Assign filter-based restrictions to string or numeric categories
    for restriction in filter_restrictions:
        for triple_info in restriction.get("path", []):
            for triple in remaining_triples:
                s, p, o = triple
                if (s.replace('ns:', '') == triple_info[0]
                        and p.replace('ns:', '') == triple_info[1]
                        and o == triple_info[2]):
                    remaining_used.add(triple)

        if restriction.get("type") == "string":
            string_restrictions.append(restriction)
        else:
            numeric_restrictions.append(restriction)

    # Collect any unclassified triples
    for triple in remaining_triples:
        if triple in remaining_used:
            continue
        s, p, o = triple
        other_restrictions.append((s, p, o))

    result = {
        "main_entity": main_entity.replace('ns:', ''),
        "main_path": [
            (s.replace('ns:', ''), p.replace('ns:', ''), o.replace('ns:', ''))
            for s, p, o in main_path
        ],
        "main_path_filters": main_path_filters,
        "entity_restrictions": [
            (s.replace('ns:', ''), p.replace('ns:', ''), o.replace('ns:', ''))
            for s, p, o in entity_restrictions
        ],
        "numeric_restrictions": numeric_restrictions,
        "string_restrictions": string_restrictions,
        "other_restrictions": [
            (s.replace('ns:', ''), p.replace('ns:', ''), o.replace('ns:', ''))
            for s, p, o in other_restrictions
        ],
        "target_variable": target_var,
    }

    return result


if __name__ == "__main__":
    import json

    # Example: a simple Freebase SPARQL query
    example_sparql = """
    PREFIX ns: <http://rdf.freebase.com/ns/>
    SELECT DISTINCT ?x
    WHERE {
        ns:m.0abc ns:people.person.nationality ?x .
    }
    """

    result = parse_sparql(example_sparql)
    if result:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print("Failed to parse the SPARQL query.")

    # Example with constraints
    example_with_filter = """
    PREFIX ns: <http://rdf.freebase.com/ns/>
    SELECT DISTINCT ?x
    WHERE {
        ns:m.0abc ns:film.actor.film ?y .
        ?y ns:film.performance.film ?x .
        ?y ns:film.performance.character ?c .
        ?c ns:type.object.name ?name .
        FILTER (str(?name) = "James Bond")
    }
    """

    result2 = parse_sparql(example_with_filter)
    if result2:
        print("\n--- Example with filter ---")
        print(json.dumps(result2, indent=2, ensure_ascii=False))
    else:
        print("Failed to parse the filtered SPARQL query.")

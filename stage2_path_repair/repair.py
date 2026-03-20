"""Stage 2 -- Path Repair Pipeline.

Loads Stage 1 CRP predictions (beam search results), checks whether each
predicted SPARQL path is reachable on Freebase, and repairs unreachable
paths via LLM-guided beam search over relations using SBERT similarity.

Typical usage::

    python -m stage2_path_repair.repair \
        --input data/cwq_beam15.json \
        --output_dir output/stage2 \
        --llm_type gpt-4o-mini \
        --api_key $OPENAI_API_KEY
"""

import argparse
import json
import logging
import os
import re
import shutil

import numpy as np
from tqdm import tqdm

# Project-local imports -- assumes the package root is on sys.path.
from utils.freebase_utils import (
    abandon_rels,
    attribute_search,
    change_paths_with_llm_cwq,
    entity_search,
    execute_sparql,
    extract_sparql_entities,
    get_embedding,
    get_entity_display_name,
    parse_structured_kbqa_string,
    reconstruct_sparql,
    replace_relation_prefix,
    select_paths_with_llm,
    sparql_all_relations,
)
from utils.llm_utils import run_llm

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# JSON I/O helpers
# ---------------------------------------------------------------------------


def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, file_path):
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


# ---------------------------------------------------------------------------
# Cosine similarity
# ---------------------------------------------------------------------------


def cosine_similarity(a, b):
    """Compute cosine similarity between vector *a* and each row of *b*.

    Returns a 1-D numpy array of similarity scores.
    """
    a = np.array(a)
    b = np.array(b)
    if b.ndim == 1:
        b = b[np.newaxis, :]
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b, axis=1)
    dot_product = np.dot(b, a)
    similarity = dot_product / (a_norm * b_norm + 1e-10)
    if similarity.size == 1:
        return np.array([similarity.item()])
    return similarity


# ---------------------------------------------------------------------------
# Relation-path string builder
# ---------------------------------------------------------------------------


def build_relation_path(relations):
    """Join a list of dotted Freebase relations into a compact path string.

    Example::

        >>> build_relation_path(["people.person.nationality", "location.country.capital"])
        "person.nationality-country.capital"
    """
    if not relations:
        return ""
    parts = relations[0].split(".")
    path = ".".join(parts[-2:])
    for rel in relations[1:]:
        parts = rel.split(".")
        path += "-" + ".".join(parts[-2:])
    return path


# ---------------------------------------------------------------------------
# Constraint entity mapping
# ---------------------------------------------------------------------------


def map_constraint_entities(pred_str, gold_entity_map, similarity_threshold=0.5):
    """Map entity names in constraint triples to gold entity MIDs.

    Mapping priority:
      1. Exact match (case-insensitive) against ``gold_entity_map`` values.
      2. Embedding-based fuzzy match (cosine similarity > *threshold*).
      3. Leave the name as-is if no match is found.

    Only processes the object of triples (text after ``]->``), and only when
    the object is *not* a variable, MID, or quoted literal.
    """
    if not gold_entity_map:
        return pred_str

    entity_label_map = {v.lower(): k for k, v in gold_entity_map.items()}

    lines = pred_str.split("\n")
    new_lines = []

    for line in lines:
        stripped = line.strip()
        match = re.search(r"(.*\]->\s+)(.+)$", stripped)
        if match:
            entity_name = match.group(2).strip()
            if (
                not entity_name.startswith("?")
                and not entity_name.startswith("m.")
                and not entity_name.startswith("g.")
                and not entity_name.startswith('"')
            ):
                mapped_mid = None

                # Priority 1: exact match
                if entity_name.lower() in entity_label_map:
                    mapped_mid = entity_label_map[entity_name.lower()]
                    logger.info("  Entity exact match: '%s' -> '%s'", entity_name, mapped_mid)
                else:
                    # Priority 2: embedding fuzzy match
                    if entity_label_map:
                        try:
                            entity_emb = get_embedding(entity_name.lower())
                            gold_names = list(entity_label_map.keys())
                            gold_embs = get_embedding(gold_names)
                            if entity_emb is not None and gold_embs is not None:
                                sims = cosine_similarity(entity_emb, np.array(gold_embs))
                                best_idx = int(np.argmax(sims))
                                best_sim = float(sims[best_idx])
                                best_name = gold_names[best_idx]
                                if best_sim > similarity_threshold:
                                    mapped_mid = entity_label_map[best_name]
                                    logger.info(
                                        "  Entity fuzzy match: '%s' -> '%s' (best: '%s', sim=%.4f)",
                                        entity_name, mapped_mid, best_name, best_sim,
                                    )
                                else:
                                    logger.info(
                                        "  Entity fuzzy match failed: '%s' (best: '%s', sim=%.4f < %.2f)",
                                        entity_name, best_name, best_sim, similarity_threshold,
                                    )
                        except Exception as exc:
                            logger.warning("  Entity embedding match error for '%s': %s", entity_name, exc)

                if mapped_mid:
                    line = line.replace(entity_name, mapped_mid)
                else:
                    logger.info("  Entity '%s' not mapped, keeping original", entity_name)

        new_lines.append(line)

    return "\n".join(new_lines)


# ---------------------------------------------------------------------------
# Direction validation
# ---------------------------------------------------------------------------


def validate_relation_direction_standalone(entity_id, relations, max_try=5):
    """Validate the direction of each hop in a relation path.

    Tries up to *max_try* intermediate entity candidates per hop to avoid
    false negatives when a single intermediate entity does not connect to the
    next hop.

    Returns:
        ``(is_valid, directions)`` where *directions* is a list of
        ``"forward"`` / ``"reverse"`` strings per relation.
    """
    current_ids = [entity_id]
    directions = []

    for rel in relations:
        found = False
        next_ids = []

        for cid in current_ids[:max_try]:
            # Try forward (head)
            head_results = entity_search(cid, rel, head=True)
            if head_results:
                if not found:
                    directions.append("forward")
                    found = True
                next_ids.extend(head_results[:3])
                continue

            # Try forward (attribute -- no successor entity)
            attr_vals = attribute_search(cid, rel)
            if attr_vals:
                if not found:
                    directions.append("forward")
                    found = True
                continue

            # Try reverse (tail)
            tail_results = entity_search(cid, rel, head=False)
            if tail_results:
                if not found:
                    directions.append("reverse")
                    found = True
                next_ids.extend(tail_results[:3])
                continue

        if not found:
            return False, directions

        if next_ids:
            current_ids = list(set(next_ids))[:max_try]

    return True, directions


# ---------------------------------------------------------------------------
# Constraint-aware path filtering
# ---------------------------------------------------------------------------


def filter_paths_by_constraints(relation_paths, initial_entities, full_parsed_result, idx):
    """Filter candidate paths by checking whether original constraints still
    produce results when combined with each candidate main path.

    For every candidate:
      1. Validate relation directions and build main-path triples.
      2. Execute SPARQL with main path only (unconstrained count).
      3. Execute SPARQL with main path + original constraints (constrained count).
      4. Keep the path if constrained results are non-empty and fewer than
         unconstrained results.

    Falls back to the original list if no path passes.
    """
    entity_restrictions = full_parsed_result.get("entity_restrictions", [])
    numeric_restrictions = full_parsed_result.get("numeric_restrictions", [])
    string_restrictions = full_parsed_result.get("string_restrictions", [])
    other_restrictions = full_parsed_result.get("other_restrictions", [])
    main_path_filters = full_parsed_result.get("main_path_filters", [])

    if not (entity_restrictions or numeric_restrictions or string_restrictions or other_restrictions):
        logger.info("Index %d: no constraints, skipping constraint filtering", idx)
        return relation_paths

    entity_id = initial_entities[0]
    logger.info("Index %d: constraint filtering over %d candidate paths ...", idx, len(relation_paths))
    passed = []

    for path_info in relation_paths:
        path_rels = path_info["path"]
        path_str = path_info.get("path_str", str(path_rels))

        try:
            # 1. Validate directions
            is_valid, directions = validate_relation_direction_standalone(entity_id, path_rels)
            if not is_valid:
                logger.debug("  %s: direction validation failed", path_str)
                continue

            # 2. Build main-path triples
            n = len(path_rels)
            if n == 4:
                variables = ["?k", "?c", "?y", "?x"]
            elif n == 3:
                variables = ["?c", "?y", "?x"]
            elif n == 2:
                variables = ["?y", "?x"]
            else:
                variables = ["?x"]

            main_path_triples = []
            current = entity_id
            for i, (rel, direction) in enumerate(zip(path_rels, directions)):
                var = variables[i]
                if direction == "forward":
                    main_path_triples.append((current, rel, var))
                else:
                    main_path_triples.append((current, f"R[{rel}]", var))
                current = var

            # 3. Unconstrained query
            no_constraint = {
                "main_entity": entity_id,
                "main_path": main_path_triples,
                "main_path_filters": [],
                "entity_restrictions": [],
                "numeric_restrictions": [],
                "string_restrictions": [],
                "other_restrictions": [],
                "target_variable": "?x",
            }
            query_no = reconstruct_sparql(no_constraint).get("simple_query", "")
            results_no = execute_sparql(query_no) if query_no else None
            entities_no = extract_sparql_entities(results_no, "x") if results_no else []

            if not entities_no:
                logger.debug("  %s: unconstrained query returned no results", path_str)
                continue

            # 4. Constrained query
            with_constraint = {
                "main_entity": entity_id,
                "main_path": main_path_triples,
                "main_path_filters": main_path_filters,
                "entity_restrictions": entity_restrictions,
                "numeric_restrictions": numeric_restrictions,
                "string_restrictions": string_restrictions,
                "other_restrictions": other_restrictions,
                "target_variable": "?x",
            }
            query_with = reconstruct_sparql(with_constraint).get("simple_query", "")
            results_with = execute_sparql(query_with) if query_with else None
            entities_with = extract_sparql_entities(results_with, "x") if results_with else []

            # 5. Decide
            if entities_with and len(entities_with) < len(entities_no):
                logger.info("  %s: constraint effective (%d < %d)", path_str, len(entities_with), len(entities_no))
                passed.append(path_info)
            else:
                logger.debug(
                    "  %s: constraint ineffective (unconstrained=%d, constrained=%d)",
                    path_str, len(entities_no), len(entities_with),
                )

        except Exception as exc:
            logger.warning("  %s: exception: %s", path_str, exc)
            continue

    if passed:
        logger.info("Index %d: constraint filter kept %d/%d paths", idx, len(passed), len(relation_paths))
        return passed

    logger.info("Index %d: no path passed constraint filter, keeping original %d paths", idx, len(relation_paths))
    return relation_paths


# ---------------------------------------------------------------------------
# LLM reasoning-path (blueprint) generation
# ---------------------------------------------------------------------------


def generate_reasoning_path(question, llm_type="gpt-4o-mini", temperature=0.4,
                            max_tokens=2048, api_key=None):
    """Use an LLM to generate a step-by-step reasoning blueprint for *question*.

    The blueprint is a list of reasoning steps (without concrete answers),
    each with optional synonymous-relation hints in parentheses.

    Returns:
        A dict ``{"reasoning_path": [step1, step2, ...]}``.
    """
    prompt = (
        "Given a question, generate the reasoning steps without providing the "
        "final answer or specific entities.\n\n"
        "Q: What countries are located in Eastern Europe and have the country calling code +373?\n"
        "A: #1 Identify the countries located in Eastern Europe.\n"
        "#2 Determine which of these countries has the country calling code +373.\n\n"
        "Q: What country that trades with Turkey has an ISO numeric code lower than 012?\n"
        "A: #1 Identify the countries that are trade partners of Turkey.\n"
        "#2 Determine which of these countries has an ISO numeric code lower than 012.\n\n"
        "Q: Who were the inspirations for the author of This Side of Paradise?\n"
        "A: #1 Identify the author of This Side of Paradise.\n"
        "#2 Determine the individuals or works that influenced the author.\n\n"
        "Q: What countries border the location where the film Amen is set?\n"
        "A: #1 Identify the location where the film Amen is set.\n"
        "#2 Determine the countries that border this location.\n\n"
        "Q: Lou Seal is the mascot for the team that last won the World Series when?\n"
        "A: #1 Identify the team for which Lou Seal is the mascot.\n"
        "#2 Determine the year this team last won the World Series.\n\n"
        f"Here is the question.\nQ: {question}\nA: "
    )

    logger.debug("Reasoning-path prompt:\n%s", prompt)

    try:
        response = run_llm(prompt, temperature, max_tokens, api_key, llm_type)
        logger.debug("LLM response:\n%s", response)

        reasoning_steps = []
        for line in response.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            if "#" in line:
                hash_index = line.index("#")
                step_content = line[hash_index + 1:].strip()
                if step_content:
                    reasoning_steps.append(step_content)

        if not reasoning_steps:
            logger.warning("No reasoning steps generated for question: '%s'", question)
            return {"reasoning_path": []}

        return {"reasoning_path": reasoning_steps}

    except Exception as exc:
        logger.error("LLM call failed during reasoning path generation: %s", exc)
        return {"reasoning_path": []}


# ---------------------------------------------------------------------------
# One-layer entity exploration (for nameless intermediate entities)
# ---------------------------------------------------------------------------


def explore_entity_one_layer(initial_entity, reasoning_steps, max_entities_per_relation=3, k=3):
    """Explore one additional hop from an intermediate entity using SBERT
    similarity between *reasoning_steps* and the entity's outgoing relations.

    Args:
        initial_entity: 6-tuple ``(entity_id, depth, parent_id, parent_rel,
            entity_name, path_relations)``.
        reasoning_steps: String describing the current reasoning step.
        max_entities_per_relation: Max neighbours to keep per relation.
        k: Number of top relations to expand.

    Returns:
        List of 6-tuples for newly discovered entities.
    """
    entity_id, entity_depth, parent_entity, parent_relation, entity_name, path_relations = initial_entity

    if not reasoning_steps:
        logger.info("Empty reasoning steps, cannot explore entity %s (ID: %s)", entity_name, entity_id)
        return []

    reasoning_embedding = get_embedding(reasoning_steps)
    if reasoning_embedding is None:
        logger.info("Reasoning embedding failed for entity %s (ID: %s), skipping", entity_name, entity_id)
        return []

    all_relations = replace_relation_prefix(
        execute_sparql(sparql_all_relations % (entity_id, entity_id))
    ) or []
    new_relations = [rel for rel in all_relations if not abandon_rels(rel)]
    if parent_relation in new_relations:
        new_relations.remove(parent_relation)
    if not new_relations:
        logger.info("Entity %s (ID: %s) has no explorable relations", entity_name, entity_id)
        return []

    relation_last_words = [".".join(rel.split(".")[-2:]) for rel in new_relations]
    relation_embeddings = get_embedding(relation_last_words)
    if relation_embeddings is None:
        logger.info("Relation embedding failed for entity %s (ID: %s), skipping", entity_name, entity_id)
        return []

    similarities = cosine_similarity(reasoning_embedding, np.array(relation_embeddings))
    relation_similarity_pairs = sorted(
        zip(new_relations, similarities), key=lambda x: x[1], reverse=True
    )
    top_relations = [rel for rel, _ in relation_similarity_pairs[:k]]

    next_entities = []
    for relation in top_relations:
        # Special handling for containment relations: head only
        if relation in ["location.location.containedby", "location.location.contains"]:
            for result in entity_search(entity_id, relation, head=True)[:max_entities_per_relation]:
                name = get_entity_display_name(result)
                if not (name.startswith("m.") or name.startswith("g.")):
                    next_entities.append((
                        result, entity_depth + 1, parent_entity,
                        parent_relation + " -> " + relation,
                        name, path_relations + [relation],
                    ))
            continue

        # Head entities
        for result in entity_search(entity_id, relation, head=True)[:max_entities_per_relation]:
            name = get_entity_display_name(result)
            if not (name.startswith("m.") or name.startswith("g.")):
                next_entities.append((
                    result, entity_depth + 1, parent_entity,
                    parent_relation + " -> " + relation,
                    name, path_relations + [relation],
                ))

        # Tail entities
        for result in entity_search(entity_id, relation, head=False)[:max_entities_per_relation]:
            name = get_entity_display_name(result)
            if not (name.startswith("m.") or name.startswith("g.")):
                next_entities.append((
                    result, entity_depth + 1, parent_entity,
                    parent_relation + " -> " + relation,
                    name, path_relations + [relation],
                ))

        # Attribute values
        for attr_value in attribute_search(entity_id, relation)[:max_entities_per_relation]:
            name = str(attr_value)
            if not (name.startswith("m.") or name.startswith("g.")):
                next_entities.append((
                    attr_value, entity_depth + 1, parent_entity,
                    parent_relation + " -> " + relation,
                    name, path_relations + [relation],
                ))

    logger.info(
        "Entity %s (ID: %s): exploration complete, found %d successor entities",
        entity_name, entity_id, len(next_entities),
    )
    return next_entities


# ---------------------------------------------------------------------------
# Triplet collection (for LLM path-selection context)
# ---------------------------------------------------------------------------


def get_triplets(current_entities, max_entities_per_relation=3):
    """Collect knowledge-graph triples reachable from *current_entities*.

    Returns:
        ``(next_entities, current_triplets)`` -- the list of newly discovered
        entity tuples and the collected ``(subject_name, relation, object_name)``
        triples.
    """
    current_triplets = []
    next_entities = []
    explored_combinations = set()

    for entity_id, entity_depth, parent_entity, parent_relation, entity_name, path_relations in current_entities:
        combination = (parent_entity, parent_relation)
        if combination in explored_combinations:
            continue
        explored_combinations.add(combination)

        parent_name = get_entity_display_name(parent_entity)

        if "-" in parent_relation:
            # Compound relation (two-hop shortcut)
            if len(path_relations) >= 2:
                last_two = path_relations[-2:]
                for head_flag in [True, False]:
                    intermediate_results = entity_search(parent_entity, last_two[0], head=head_flag)
                    for inter in intermediate_results[:max_entities_per_relation]:
                        final_results = (
                            entity_search(inter, last_two[1], head=False)
                            + entity_search(inter, last_two[1], head=True)
                        )
                        for final_ent in final_results[:max_entities_per_relation]:
                            name = get_entity_display_name(final_ent)
                            if not (name.startswith("m.") or name.startswith("g.")):
                                current_triplets.append((parent_name, parent_relation, name))
                                next_entities.append((
                                    final_ent, entity_depth, parent_entity,
                                    parent_relation, str(name), path_relations,
                                ))
                        for attr in attribute_search(inter, last_two[1])[:max_entities_per_relation]:
                            current_triplets.append((parent_name, parent_relation, attr))
        else:
            # Single relation
            for attr in attribute_search(parent_entity, parent_relation)[:max_entities_per_relation]:
                current_triplets.append((parent_name, parent_relation, attr))
            for result in entity_search(parent_entity, parent_relation, head=True)[:max_entities_per_relation]:
                if result != parent_entity:
                    name = get_entity_display_name(result)
                    if not (name.startswith("m.") or name.startswith("g.")):
                        current_triplets.append((parent_name, parent_relation, name))
                        next_entities.append((
                            result, entity_depth, parent_entity,
                            parent_relation, str(name), path_relations,
                        ))
            for result in entity_search(parent_entity, parent_relation, head=False)[:max_entities_per_relation]:
                if result != parent_entity:
                    name = get_entity_display_name(result)
                    if not (name.startswith("m.") or name.startswith("g.")):
                        current_triplets.append((parent_name, parent_relation, name))
                        next_entities.append((
                            result, entity_depth, parent_entity,
                            parent_relation, str(name), path_relations,
                        ))

    return next_entities, current_triplets


# ---------------------------------------------------------------------------
# Helper: convert dotted relation to comma-separated display form
# ---------------------------------------------------------------------------


def _convert_to_comma_separated(rel):
    """Convert ``'people.person.profession'`` to ``'people , person , profession'``."""
    parts = rel.split(".")
    formatted = [part.replace("_", " ") for part in parts]
    return " , ".join(formatted)


# ---------------------------------------------------------------------------
# Main exploration / repair pipeline
# ---------------------------------------------------------------------------


def explore_freebase(
    input_file,
    output_dir,
    llm_type="gpt-4o-mini",
    api_key=None,
    depth=2,
    k=4,
    m=10,
    beam_width=3,
    max_entities_per_relation=5,
    batch_size=10,
    triplets_number=15,
):
    """Run the Stage 2 path-repair pipeline.

    For each question in *input_file*:
      1. Parse the top CRP prediction from Stage 1.
      2. Map the predicted topic entity to the closest gold entity.
      3. Reconstruct SPARQL and test reachability.
      4. If reachable -- skip (Stage 1 succeeded).
      5. Otherwise -- generate a reasoning blueprint via LLM, perform beam
         search over Freebase relations using SBERT similarity, select paths
         via LLM, validate relation directions, and replace the main path
         in the CRP prediction.
      6. Save updated predictions.

    Args:
        input_file: Path to Stage 1 predictions JSON.
        output_dir: Directory for output files and logs.
        llm_type: LLM model identifier (default ``"gpt-4o-mini"``).
        api_key: API key for the LLM service.
        depth: Maximum beam-search depth (default 2).
        k: Number of top relations to keep per reasoning step (default 4).
        m: Number of top candidate paths to keep (default 10).
        beam_width: Beam width -- number of paths selected by LLM at
            intermediate depths (default 3).
        max_entities_per_relation: Max entity results per relation query.
        batch_size: Items per checkpoint save.
        triplets_number: Max entities per relation when collecting triples
            for LLM context.
    """
    os.makedirs(output_dir, exist_ok=True)

    logger.info("Reading input file: %s", input_file)
    cwq_data = load_json(input_file)
    logger.info("Loaded %d items", len(cwq_data))

    input_basename = os.path.splitext(os.path.basename(input_file))[0]
    temp_file = os.path.join(output_dir, f"{input_basename}_temp.json")
    final_file = os.path.join(output_dir, f"{input_basename}_with_reasoning.json")
    total_items = len(cwq_data)

    for start_idx in range(0, total_items, batch_size):
        end_idx = min(start_idx + batch_size, total_items)
        logger.info("Processing batch %d to %d (of %d) ...", start_idx, end_idx - 1, total_items)

        for idx in tqdm(range(start_idx, end_idx), desc="Processing"):
            cwq_item = cwq_data[idx]
            question = cwq_item.get("question", "")
            if not question:
                logger.info("Index %d: missing 'question' field, skipping", idx)
                continue

            topic_entity_dict = cwq_item.get("gold_entity_map", {})
            initial_entities = list(topic_entity_dict.keys())

            # ------------------------------------------------------------------
            # Map the predicted entity in the first prediction to the closest
            # gold entity via embedding similarity.
            # ------------------------------------------------------------------
            predictions = cwq_item.get("pred", {}).get("predictions", [])
            if predictions:
                first_pred = predictions[0]
                try:
                    first_line = first_pred.split("\n")[1]
                    first_entity = first_line.split("-[")[0].strip(" ()")
                    if first_entity and topic_entity_dict:
                        gold_values = list(topic_entity_dict.values())
                        gold_keys = list(topic_entity_dict.keys())
                        pred_emb = get_embedding(first_entity)
                        gold_embs = get_embedding(gold_values)
                        if pred_emb is not None and gold_embs is not None:
                            sims = cosine_similarity(pred_emb, np.array(gold_embs))
                            best_idx_val = int(np.argmax(sims))
                            best_gold_id = gold_keys[best_idx_val]
                            logger.info(
                                "Index %d: predicted entity '%s' -> gold '%s' (%s)",
                                idx, first_entity, gold_values[best_idx_val], best_gold_id,
                            )
                            initial_entities = [best_gold_id]
                except Exception as exc:
                    logger.warning("Index %d: failed to extract predicted entity: %s", idx, exc)

            # ------------------------------------------------------------------
            # Parse the CRP prediction and check SPARQL reachability.
            # ------------------------------------------------------------------
            full_parsed_result = None
            main_paths = []
            first_pred_text = ""
            filtered_result = None

            if predictions:
                try:
                    first_pred_text = predictions[0]

                    # Replace the first entity in the main path with the
                    # mapped gold entity.
                    if initial_entities:
                        lines = first_pred_text.split("\n")
                        for i, line in enumerate(lines):
                            if line.strip() and not line.strip().endswith(":"):
                                if "-[" in line:
                                    left_entity = line.split("-[", 1)[0].strip(" ()")
                                    lines[i] = line.replace(left_entity, initial_entities[0], 1)
                                    logger.info(
                                        "Index %d: replaced entity '%s' -> '%s'",
                                        idx, left_entity, initial_entities[0],
                                    )
                                    break
                        first_pred_text = "\n".join(lines)

                    # Map constraint entity names to MIDs
                    first_pred_text = map_constraint_entities(first_pred_text, topic_entity_dict)

                    parsed_result = parse_structured_kbqa_string(first_pred_text)
                    logger.debug("Index %d: parsed CRP: %s", idx, parsed_result)
                    full_parsed_result = parsed_result

                    # Build a query using only the main path (ignore constraints
                    # for reachability check).
                    filtered_result = {
                        "main_entity": parsed_result.get("main_entity", ""),
                        "main_path": parsed_result.get("main_path", []),
                        "main_path_filters": parsed_result.get("main_path_filters", []),
                        "entity_restrictions": [],
                        "numeric_restrictions": [],
                        "string_restrictions": [],
                        "other_restrictions": [],
                        "target_variable": "",
                    }

                    sparql_queries = reconstruct_sparql(filtered_result)
                    simple_query = sparql_queries.get("simple_query", "")

                    if simple_query:
                        logger.info("Index %d: executing reachability SPARQL ...", idx)
                        query_results = execute_sparql(simple_query)
                        entity_ids = extract_sparql_entities(query_results, "x")
                        logger.info("Index %d: reachability results: %s", idx, entity_ids)

                        if entity_ids:
                            logger.info(
                                "Index %d: reachable (%d entities), skipping repair",
                                idx, len(entity_ids),
                            )
                            continue

                        # Extract relation list from main-path triples
                        path_relations = []
                        for triplet in filtered_result.get("main_path", []):
                            if isinstance(triplet, (tuple, list)) and len(triplet) >= 2:
                                rel = triplet[1]
                                if rel.startswith("R[") and rel.endswith("]"):
                                    rel = rel[2:-1]
                                path_relations.append(rel)
                            elif isinstance(triplet, str) and "-[" in triplet and "]->" in triplet:
                                path_relations.append(triplet.split("-[")[1].split("]->")[0])

                        # Skip paths containing the noisy profession relation
                        if "people.person.profession" in path_relations:
                            logger.info("Index %d: main path contains noisy relation, skipping", idx)
                            continue

                        path_str = build_relation_path(path_relations)

                        # No reachability -- record the main path with empty marker
                        main_paths.append({
                            "path": path_relations,
                            "path_str": path_str,
                            "similarity": None,
                            "end_entity": ["<EMPTY_RESULT>"],
                        })
                        logger.info("Index %d: main path (unreachable): %s", idx, main_paths)

                except Exception as exc:
                    logger.warning("Index %d: SPARQL processing failed: %s", idx, exc)

            # ------------------------------------------------------------------
            # Generate reasoning blueprint
            # ------------------------------------------------------------------
            reasoning_result = generate_reasoning_path(
                question, llm_type=llm_type, temperature=0, max_tokens=2048, api_key=api_key,
            )
            reasoning_steps = reasoning_result.get("reasoning_path", [])
            cwq_item["reasoning_path"] = reasoning_steps
            if not reasoning_steps:
                logger.info("Index %d: reasoning path is empty, skipping", idx)
                continue

            question_embedding = get_embedding(question)
            if question_embedding is None:
                logger.info("Index %d: question embedding failed, skipping", idx)
                continue

            logger.info("Index %d: reasoning steps: %s", idx, reasoning_steps)
            reasoning_embeddings = get_embedding(reasoning_steps)
            if reasoning_embeddings is None or (
                hasattr(reasoning_embeddings, "__iter__")
                and any(emb is None for emb in reasoning_embeddings)
            ):
                logger.info("Index %d: reasoning embedding failed, skipping", idx)
                continue
            valid_reasoning_embeddings = np.array(reasoning_embeddings)
            valid_reasoning_steps = reasoning_steps

            # ------------------------------------------------------------------
            # Beam-search over relations
            # ------------------------------------------------------------------
            visited_entities = []
            entity_history = {}
            selected_path = []
            current_entities = [
                (eid, 0, None, None, get_entity_display_name(eid), [])
                for eid in initial_entities
            ]

            # Dynamic depth calculation based on main-path length
            calculated_depth = depth
            if filtered_result is not None:
                main_path_triplets = filtered_result.get("main_path", [])
                num_triplets = len(main_path_triplets)
                if num_triplets == 1:
                    calculated_depth = 1
                elif num_triplets == 2:
                    has_y = any("?y" in str(t) for t in main_path_triplets)
                    if has_y:
                        calculated_depth = 1

            logger.info("Index %d: computed search depth: %d", idx, calculated_depth)

            for current_depth in range(calculated_depth):
                logger.info(
                    "Index %d: depth %d, %d entities to explore",
                    idx, current_depth, len(current_entities),
                )
                if not current_entities:
                    logger.info("Index %d: depth %d -- no entities to explore, stopping", idx, current_depth)
                    break

                # Collect all relations from current entities
                all_relations = []
                for current_entity, _, parent_entity, parent_relation, _, _ in current_entities:
                    if parent_entity and parent_relation:
                        entity_history[current_entity] = (parent_entity, parent_relation)
                    tem_relations = replace_relation_prefix(
                        execute_sparql(sparql_all_relations % (current_entity, current_entity))
                    ) or []
                    all_relations += [rel for rel in tem_relations if not abandon_rels(rel)]

                # De-duplicate while preserving order
                final_relation_map = {
                    rel: ".".join(rel.split(".")[-2:]) for rel in all_relations
                }
                if not final_relation_map:
                    logger.info("Index %d: no relations found at depth %d", idx, current_depth)
                    all_embeddings = np.array([])
                    pre_unique_relations = []
                else:
                    relation_last_words = list(final_relation_map.values())
                    all_embeddings = get_embedding(relation_last_words)
                    all_relations = list(final_relation_map.keys())
                    pre_unique_relations = list(zip(all_relations, all_embeddings))
                    all_embeddings = np.array(all_embeddings)

                # Assign relations to reasoning steps by highest similarity
                path_relation_assignments = [[] for _ in valid_reasoning_steps]
                for rel, emb in pre_unique_relations:
                    similarities = cosine_similarity(emb, valid_reasoning_embeddings)
                    best_path_idx = int(np.argmax(similarities))
                    path_relation_assignments[best_path_idx].append((rel, similarities[best_path_idx]))

                # For each reasoning step, keep the top-2k relations
                all_top_k_relations = []
                for path_idx, assignments in enumerate(path_relation_assignments):
                    if assignments:
                        assignments.sort(key=lambda x: x[1], reverse=True)
                        top_count = min(2 * k, len(assignments))
                        all_top_k_relations.append((path_idx, assignments[:top_count]))

                # Query next-hop entities for top relations
                next_entities = []
                entity_similarity_map = {}

                for path_idx, path_relations_list in all_top_k_relations:
                    for current_entity, _, parent_entity, parent_relation, entity_name, _ in current_entities:
                        for rel, similarity in path_relations_list:
                            # Special handling for containment relations
                            if rel in ["location.location.contains", "location.location.containedby"]:
                                head_results = entity_search(current_entity, rel, head=True)
                                for result in head_results[:max_entities_per_relation]:
                                    if result not in [e[0] for e in visited_entities]:
                                        next_entities.append((
                                            result, current_depth + 1, current_entity, rel,
                                            get_entity_display_name(result), [],
                                        ))
                                        ekey = f"{result}_{path_idx}"
                                        if ekey not in entity_similarity_map or entity_similarity_map[ekey][1] < similarity:
                                            entity_similarity_map[ekey] = (result, similarity, rel, path_idx)
                                continue

                            # Head entities
                            for result in entity_search(current_entity, rel, head=True)[:max_entities_per_relation]:
                                if result not in [e[0] for e in visited_entities]:
                                    next_entities.append((
                                        result, current_depth + 1, current_entity, rel,
                                        get_entity_display_name(result), [],
                                    ))
                                    ekey = f"{result}_{path_idx}"
                                    if ekey not in entity_similarity_map or entity_similarity_map[ekey][1] < similarity:
                                        entity_similarity_map[ekey] = (result, similarity, rel, path_idx)

                            # Tail entities
                            for result in entity_search(current_entity, rel, head=False)[:max_entities_per_relation]:
                                if result not in [e[0] for e in visited_entities]:
                                    next_entities.append((
                                        result, current_depth + 1, current_entity, rel,
                                        get_entity_display_name(result), [],
                                    ))
                                    ekey = f"{result}_{path_idx}"
                                    if ekey not in entity_similarity_map or entity_similarity_map[ekey][1] < similarity:
                                        entity_similarity_map[ekey] = (result, similarity, rel, path_idx)

                            # Attribute values
                            for attr_value in attribute_search(current_entity, rel)[:max_entities_per_relation]:
                                if attr_value not in [e[0] for e in visited_entities]:
                                    next_entities.append((
                                        attr_value, current_depth + 1, current_entity, rel,
                                        str(attr_value), [],
                                    ))
                                    ekey = f"{attr_value}_{path_idx}"
                                    if ekey not in entity_similarity_map or entity_similarity_map[ekey][1] < similarity:
                                        entity_similarity_map[ekey] = (attr_value, similarity, rel, path_idx)

                # ---------------------------------------------------------------
                # Filter next entities: keep only those whose relation is in the
                # top-k for the assigned reasoning step.
                # ---------------------------------------------------------------
                path_to_similarity_map = {}
                for ekey, (ent, sim, rel, pidx) in entity_similarity_map.items():
                    path_to_similarity_map.setdefault(pidx, []).append((sim, rel))

                path_top_k_relations_map = {}
                for pidx, sim_rel_pairs in path_to_similarity_map.items():
                    sim_rel_pairs.sort(reverse=True)
                    top_k_rels = []
                    seen_sims = set()
                    for sim, rel in sim_rel_pairs:
                        if sim not in seen_sims and len(top_k_rels) < k:
                            top_k_rels.append(rel)
                            seen_sims.add(sim)
                        if len(top_k_rels) == k:
                            break
                    path_top_k_relations_map[pidx] = top_k_rels

                filtered_next_entities = []
                for entity_data in next_entities:
                    _, _, _, parent_rel, _, _ = entity_data
                    ent_id = entity_data[0]
                    for pidx in range(len(reasoning_steps)):
                        ekey = f"{ent_id}_{pidx}"
                        if ekey in entity_similarity_map:
                            _, _, _, stored_pidx = entity_similarity_map[ekey]
                            if parent_rel in path_top_k_relations_map.get(stored_pidx, []):
                                filtered_next_entities.append(entity_data)

                visited_entities = current_entities
                current_entities = filtered_next_entities

                # ---------------------------------------------------------------
                # Build relation paths and compute similarities
                # ---------------------------------------------------------------
                relation_paths = []
                path_str_set = set()
                next_entities_to_explore = []
                updated_entities = []

                for entity_id_c, entity_depth_c, parent_entity_c, parent_relation_c, entity_name_c, path_rels_c in current_entities:
                    new_paths = []
                    if parent_entity_c:
                        parent_data_list = [e for e in visited_entities if e[0] == parent_entity_c]
                        if parent_data_list and not path_rels_c:
                            for pd in parent_data_list:
                                new_paths.append(pd[5] + [parent_relation_c])
                    path_rels_list = new_paths if new_paths else [path_rels_c]

                    for pth in path_rels_list:
                        if (
                            isinstance(entity_id_c, str)
                            and (entity_id_c.startswith("m.") or entity_id_c.startswith("g."))
                            and (entity_name_c.startswith("m.") or entity_name_c.startswith("g."))
                        ):
                            next_entities_to_explore.append((
                                entity_id_c, entity_depth_c, parent_entity_c,
                                parent_relation_c, entity_name_c, pth,
                            ))
                        else:
                            updated_entities.append((
                                entity_id_c, entity_depth_c, parent_entity_c,
                                parent_relation_c, entity_name_c, pth,
                            ))

                # De-duplicate exploration candidates
                next_entities_to_explore = [
                    (eid, ed, pe, pr, en, list(p))
                    for eid, ed, pe, pr, en, p in set(
                        (eid, ed, pe, pr, en, tuple(p))
                        for eid, ed, pe, pr, en, p in next_entities_to_explore
                    )
                ]

                # Explore nameless intermediate entities one more hop
                for entity_data in tqdm(next_entities_to_explore, desc="Exploring intermediates", leave=False):
                    eid = entity_data[0]
                    for pidx in range(len(reasoning_steps)):
                        ekey = f"{eid}_{pidx}"
                        if ekey in entity_similarity_map:
                            _, _, _, stored_pidx = entity_similarity_map[ekey]
                            explored = explore_entity_one_layer(
                                entity_data, reasoning_steps[stored_pidx],
                                max_entities_per_relation, 3,
                            )
                            updated_entities.extend(explored)

                current_entities = [
                    (eid, ed, pe, pr, en, pr_list)
                    for eid, ed, pe, pr, en, pr_list in updated_entities
                ]

                # Build path strings and compute question-path similarity
                for eid, _, _, _, _, path_rels_c in current_entities:
                    if path_rels_c:
                        ps = build_relation_path(path_rels_c)
                        if ps not in path_str_set:
                            path_str_set.add(ps)
                            path_emb = get_embedding(ps + get_entity_display_name(eid))
                            if path_emb is not None:
                                sim = cosine_similarity(path_emb, question_embedding)
                                relation_paths.append({
                                    "path": path_rels_c,
                                    "path_str": ps,
                                    "similarity": float(sim),
                                    "end_entity": [eid],
                                })
                        else:
                            for rp in relation_paths:
                                if rp["path_str"] == ps:
                                    if eid not in rp["end_entity"]:
                                        rp["end_entity"].append(eid)
                                    break

                # Filter out degenerate paths
                filtered_relation_paths = []
                for rp in relation_paths:
                    end_ents = rp["end_entity"]
                    if len(end_ents) == 1 and end_ents[0] in initial_entities:
                        continue
                    if any(
                        not (e.startswith("m.") or e.startswith("g."))
                        for e in end_ents
                        if isinstance(e, str)
                    ):
                        continue
                    filtered_relation_paths.append(rp)
                relation_paths = filtered_relation_paths

                # Constraint filtering at the last depth level
                if current_depth == calculated_depth - 1 and full_parsed_result and relation_paths:
                    relation_paths = filter_paths_by_constraints(
                        relation_paths, initial_entities, full_parsed_result, idx,
                    )

                # Keep top-m paths by similarity
                relation_paths.sort(key=lambda x: x["similarity"], reverse=True)
                top_m_paths = relation_paths[:m]

                # ---------------------------------------------------------------
                # LLM path selection / replacement
                # ---------------------------------------------------------------
                top_m_relations = []

                if calculated_depth == 2 and current_depth != 1:
                    # Intermediate depth: select paths via LLM, then continue
                    top_m_relations = select_paths_with_llm(
                        question, top_m_paths, initial_entities, [],
                        llm_type, 0, 2048, api_key,
                    )
                    current_entities = [
                        (eid, ed, pe, pr, en, pr_list)
                        for eid, ed, pe, pr, en, pr_list in current_entities
                        if pr_list in top_m_relations and isinstance(eid, str)
                    ]
                    current_entities, _ = get_triplets(current_entities, triplets_number)
                    current_entities = [
                        (eid, ed, pe, pr, en, pr_list)
                        for eid, ed, pe, pr, en, pr_list in current_entities
                        if isinstance(eid, str) and eid.startswith("m.")
                    ]
                    continue

                # Final depth: decide whether to replace the main path
                if not main_paths:
                    logger.info("Index %d: no main path available for replacement", idx)
                    break

                selected_path_obj, should_replace = change_paths_with_llm_cwq(
                    question, top_m_paths, main_paths[0], initial_entities,
                    selected_path if selected_path else None,
                    llm_type, 0, 2048, api_key,
                )
                top_m_relations.append(selected_path_obj["path"])

                if not should_replace:
                    logger.info("Index %d: keeping original main path", idx)
                    save_json(cwq_data, temp_file)
                    break

                if calculated_depth == 2 and current_depth != 1:
                    logger.info("Index %d: continuing to depth 2", idx)
                else:
                    # Replace the main path in the CRP prediction
                    path_relations_sel = selected_path_obj.get("path", [])
                    is_valid, directions = validate_relation_direction_standalone(
                        initial_entities[0], path_relations_sel,
                    )

                    if is_valid:
                        lines = first_pred_text.split("\n")
                        main_path_start = None
                        main_path_end = len(lines)
                        for i, line in enumerate(lines):
                            if line.strip().startswith("Main Path:"):
                                main_path_start = i
                            elif main_path_start is not None and line.strip().endswith(":"):
                                main_path_end = i
                                break

                        if main_path_start is not None:
                            new_main_path_lines = ["Main Path:"]
                            current_ent = initial_entities[0]
                            n_rels = len(path_relations_sel)
                            if n_rels == 4:
                                variables = ["?k", "?c", "?y", "?x"]
                            elif n_rels == 3:
                                variables = ["?c", "?y", "?x"]
                            elif n_rels == 2:
                                variables = ["?y", "?x"]
                            else:
                                variables = ["?x"]

                            for j, rel in enumerate(path_relations_sel):
                                d = directions[j]
                                rel_str = rel if d == "forward" else f"R[{rel}]"
                                var = variables[j] if j < len(variables) else "?x"
                                new_main_path_lines.append(f"{current_ent} -[{rel_str}]-> {var}")
                                current_ent = var

                            lines = lines[:main_path_start] + new_main_path_lines + lines[main_path_end:]
                            first_pred_text = "\n".join(lines)
                            logger.info("Index %d: replaced main path -> %s", idx, first_pred_text)

                            # Re-execute SPARQL to verify the replacement
                            parsed_new = parse_structured_kbqa_string(first_pred_text)
                            filtered_new = {
                                "main_entity": parsed_new.get("main_entity", ""),
                                "main_path": parsed_new.get("main_path", []),
                                "main_path_filters": parsed_new.get("main_path_filters", []),
                                "entity_restrictions": [],
                                "numeric_restrictions": [],
                                "string_restrictions": [],
                                "other_restrictions": [],
                                "target_variable": "",
                            }
                            new_query = reconstruct_sparql(filtered_new).get("simple_query", "")
                            if new_query:
                                new_results = execute_sparql(new_query)
                                new_entity_ids = extract_sparql_entities(new_results, "x")
                                logger.info("Index %d: post-replacement results: %s", idx, new_entity_ids)

                                if new_entity_ids:
                                    # Convert MIDs to display names in the CRP text
                                    initial_name = get_entity_display_name(initial_entities[0])
                                    pred_lines = first_pred_text.split("\n")
                                    for li, line in enumerate(pred_lines):
                                        if line.strip() and not line.strip().endswith(":") and "-[" in line:
                                            parts = line.split("-[", 1)
                                            if len(parts) == 2:
                                                ent_part = parts[0].strip(" ()")
                                                rel_target = parts[1].split("]->", 1)
                                                if len(rel_target) == 2:
                                                    rel_part = rel_target[0].strip()
                                                    target_part = rel_target[1].strip()
                                                    if ent_part.startswith("m.") or ent_part.startswith("g."):
                                                        ent_part = initial_name
                                                    formatted_rel = _convert_to_comma_separated(rel_part)
                                                    pred_lines[li] = f"{ent_part} -[{formatted_rel}]-> {target_part}"
                                    first_pred_text = "\n".join(pred_lines)

                                    cwq_item["pred"]["predictions"].insert(0, first_pred_text)
                                    save_json(cwq_data, temp_file)
                                    logger.info(
                                        "Index %d: inserted repaired prediction and saved checkpoint",
                                        idx,
                                    )
                        else:
                            logger.warning("Index %d: 'Main Path:' section not found in prediction", idx)
                    else:
                        logger.info("Index %d: direction validation failed, keeping original path", idx)

                    cwq_item["selected_path"] = selected_path_obj
                    save_json(cwq_data, temp_file)
                    break

                # Prepare entities for the next depth level
                current_entities = [
                    (eid, ed, pe, pr, en, pr_list)
                    for eid, ed, pe, pr, en, pr_list in current_entities
                    if pr_list in top_m_relations and isinstance(eid, str)
                ]
                current_entities, _ = get_triplets(current_entities, triplets_number)
                current_entities = [
                    (eid, ed, pe, pr, en, pr_list)
                    for eid, ed, pe, pr, en, pr_list in current_entities
                    if isinstance(eid, str) and eid.startswith("m.")
                ]

        save_json(cwq_data, temp_file)
        logger.info("Batch %d-%d saved to %s", start_idx, end_idx - 1, temp_file)

    shutil.move(temp_file, final_file)
    logger.info("Processing complete. Output saved to %s", final_file)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Stage 2: Path Repair -- fix unreachable CRP predictions via "
                    "LLM-guided beam search over Freebase relations.",
    )
    parser.add_argument(
        "--input", type=str, required=True,
        help="Path to Stage 1 predictions JSON file.",
    )
    parser.add_argument(
        "--output_dir", type=str, default="output/stage2",
        help="Directory for output files and logs (default: output/stage2).",
    )
    parser.add_argument(
        "--llm_type", type=str, default="gpt-4o-mini",
        help="LLM model identifier (default: gpt-4o-mini).",
    )
    parser.add_argument(
        "--api_key", type=str, default=None,
        help="API key for the LLM service. Falls back to OPENAI_API_KEY env var.",
    )
    parser.add_argument(
        "--k", type=int, default=4,
        help="Number of top relations to keep per reasoning step (default: 4).",
    )
    parser.add_argument(
        "--m", type=int, default=10,
        help="Number of top candidate paths to retain (default: 10).",
    )
    parser.add_argument(
        "--beam_width", type=int, default=3,
        help="Beam width for intermediate-depth LLM selection (default: 3).",
    )
    parser.add_argument(
        "--batch_size", type=int, default=10,
        help="Checkpoint save frequency in items (default: 10).",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable verbose (DEBUG level) logging.",
    )
    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        parser.error("An API key is required. Provide --api_key or set OPENAI_API_KEY.")

    explore_freebase(
        input_file=args.input,
        output_dir=args.output_dir,
        llm_type=args.llm_type,
        api_key=api_key,
        depth=2,
        k=args.k,
        m=args.m,
        beam_width=args.beam_width,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()

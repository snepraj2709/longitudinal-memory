#!/usr/bin/env python3
"""Generate the deterministic Step R2 scaled benchmark release."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any


BENCHMARK_VERSION = "scaled_v1"
CAPABILITIES = (
    "extraction",
    "temporal_reasoning",
    "conflict_detection",
    "user_modeling",
    "abstention",
)

PEOPLE = (
    ("Asha", "Riverstone Labs", "product engineer", "data products", "build reliable health tools", "Nikhil", "Leena", "Pune", "Mumbai", "Care Map", "Kabir"),
    ("Mateo", "Northstar Studio", "design researcher", "service design", "improve public-service research", "Sofia", "Elias", "Madrid", "Valencia", "Civic Lens", "Lucia"),
    ("Priya", "Cedar Robotics", "software engineer", "robotics platforms", "lead dependable robotics systems", "Arjun", "Meera", "Bengaluru", "Chennai", "Fleet Signal", "Rohan"),
    ("Jonah", "Harbor Health", "operations analyst", "care operations", "make clinic operations less wasteful", "Mara", "Theo", "Boston", "Providence", "Clinic Flow", "Nora"),
    ("Lin", "Brightfield Energy", "policy analyst", "energy policy", "shape practical clean-energy programs", "Wei", "Amara", "Singapore", "Kuala Lumpur", "Grid Access", "Jun"),
    ("Samira", "Mosaic Learning", "curriculum designer", "learning design", "build accessible science courses", "Omar", "Tara", "Dubai", "Abu Dhabi", "Open Lab", "Zain"),
    ("Noah", "Atlas Finance", "risk engineer", "financial systems", "improve transparent risk tooling", "Iris", "Caleb", "Toronto", "Ottawa", "Clear Risk", "Mila"),
    ("Elena", "Willow Foods", "supply planner", "food systems", "reduce waste across supply planning", "Rafael", "Ines", "Lisbon", "Porto", "Fresh Route", "Tiago"),
    ("Darius", "Openway Transit", "data analyst", "transport analytics", "make transit planning more equitable", "Imani", "Grace", "Chicago", "Detroit", "Route Equity", "Avery"),
    ("Keiko", "Lumen Media", "audience strategist", "media research", "develop trustworthy audience research", "Hana", "Ren", "Tokyo", "Osaka", "Signal Study", "Yuki"),
)


def _record(path: str, layer: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    payload = "".join(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n" for item in records)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(payload, encoding="utf-8")
    return {
        "path": path,
        "layer": layer,
        "sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        "records": len(records),
    }


def _json_file(path: str, layer: str, value: dict[str, Any]) -> dict[str, Any]:
    payload = json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(payload, encoding="utf-8")
    return {
        "path": path,
        "layer": layer,
        "sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        "records": 1,
    }


def _dataset_hash(files: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for item in sorted(files, key=lambda value: value["path"]):
        path = item["path"]
        content = Path(path).read_bytes()
        encoded = path.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def _message(source_id: str, number: int, speaker: str, text: str) -> dict[str, Any]:
    return {
        "message_id": f"{source_id}_message_{number:03d}",
        "speaker_id": speaker,
        "text": text,
    }


def _evidence(source: dict[str, Any], quote: str, message: int | None = 1) -> dict[str, Any]:
    return {
        "source_id": source["source_id"],
        "message_id": None if message is None else source["messages"][message - 1]["message_id"],
        "quote": quote,
    }


def _source(
    user_id: str,
    number: int,
    source_type: str,
    timestamp: str,
    participants: list[str],
    texts: list[tuple[str, str]],
    title: str | None = None,
) -> dict[str, Any]:
    source_id = f"scaled_{user_id}_{source_type}_{number:03d}"
    messages = [] if source_type == "calendar" else [
        _message(source_id, index, speaker, text)
        for index, (speaker, text) in enumerate(texts, 1)
    ]
    content = texts[0][1] if source_type == "calendar" else "\n".join(
        f"{speaker}: {text}" for speaker, text in texts
    )
    return {
        "source_id": source_id,
        "source_type": source_type,
        "user_id": user_id,
        "created_at": timestamp,
        "ingested_at": timestamp,
        "participants": participants,
        "content": content,
        "messages": messages,
        "metadata": {"title": title, "thread_id": f"{source_id}_thread"},
    }


def _common_case(
    case_id: str,
    user_id: str,
    split: str,
    task: str,
    capability: str,
    difficulty: str,
) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "benchmark_version": BENCHMARK_VERSION,
        "split": split,
        "user_id": user_id,
        "task": task,
        "capability": capability,
        "as_of": "2026-12-01T12:00:00+00:00",
        "difficulty": difficulty,
    }


def _review(queue: str, user_id: str, target_type: str, target_id: str, checks: list[str]) -> dict[str, Any]:
    return {
        "review_id": f"review_{queue}_{target_id}",
        "benchmark_version": BENCHMARK_VERSION,
        "queue": queue,
        "user_id": user_id,
        "target_type": target_type,
        "target_id": target_id,
        "status": "pending_human_review",
        "checks": checks,
        "notes": "",
    }


def _build_user(index: int, values: tuple[str, ...]) -> dict[str, list[dict[str, Any]]]:
    name, company, role, initial_goal, current_goal, manager, mentor, city, possible_city, project, other_person = values
    user_id = f"user_{index:03d}"
    split = "development" if index <= 2 else "test"
    wrong_date = f"2026-02-{index + 2:02d}"
    correct_date = f"2026-01-{index + 10:02d}"
    milestone_date = f"2026-08-{index + 10:02d}"
    uncertain_date = f"2026-12-{index + 5:02d}"

    quotes = {
        "intro": f"I accepted the {role} role at {company}. For now, I want to deepen my work in {initial_goal}. I do my best focused work remotely.",
        "wrong_date": f"Our onboarding sheet records {name}'s start date as {wrong_date}.",
        "wrong_person": f"{other_person} is moving to {possible_city}; this is about {other_person}, not {name}.",
        "move": f"Planning discussion only. {name} has not decided to move to {possible_city}.",
        "shift": f"This month I feel exhausted after the launch. My goal has shifted: I want to {current_goal}.",
        "assignment": f"{name} will lead {project}. {mentor} will mentor {name} on the work.",
        "office_one": f"I heard {name}'s office base is {city}.",
        "office_two": f"My roster lists {name} in {possible_city}, but it may be stale.",
        "milestone": f"{project} review is scheduled for {milestone_date}.",
        "correction": f"I need to correct the start date. I began on {correct_date}, not {wrong_date}. My current goal is still to {current_goal}.",
        "closing": f"Remote work has suited me throughout the year. I might move to {possible_city} someday, but I have made no plan. The deadline may be {uncertain_date}, though I need to confirm it.",
    }

    sources = [
        _source(user_id, 1, "conversation", "2026-01-08T09:00:00+00:00", [user_id, manager.lower()], [(user_id, quotes["intro"])]),
        _source(user_id, 1, "email", "2026-02-05T10:00:00+00:00", [user_id, "recruiting"], [("recruiting", quotes["wrong_date"])]),
        _source(user_id, 1, "chat", "2026-03-12T11:00:00+00:00", [user_id, other_person.lower()], [(other_person.lower(), quotes["wrong_person"])]),
        _source(user_id, 1, "calendar", "2026-04-15T12:00:00+00:00", [user_id], [(user_id, quotes["move"])], "Relocation planning"),
        _source(user_id, 2, "conversation", "2026-05-18T09:30:00+00:00", [user_id, mentor.lower()], [(user_id, quotes["shift"])]),
        _source(user_id, 2, "email", "2026-06-20T10:30:00+00:00", [user_id, manager.lower(), mentor.lower()], [(manager.lower(), quotes["assignment"])]),
        _source(user_id, 2, "chat", "2026-07-22T11:30:00+00:00", [user_id, manager.lower(), "teammate"], [("teammate", quotes["office_one"]), (manager.lower(), quotes["office_two"])]),
        _source(user_id, 2, "calendar", "2026-08-01T08:00:00+00:00", [user_id, manager.lower()], [(user_id, quotes["milestone"])], f"{project} review"),
        _source(user_id, 3, "conversation", "2026-09-10T09:00:00+00:00", [user_id, manager.lower()], [(user_id, quotes["correction"])]),
        _source(user_id, 4, "conversation", "2026-11-14T09:00:00+00:00", [user_id, mentor.lower()], [(user_id, quotes["closing"])]),
    ]
    # Stable aliases make the templates below readable.
    conv1, email1, chat1, cal1, conv2, email2, chat2, cal2, conv3, conv4 = sources

    users = [{
        "user_id": user_id,
        "display_name": name,
        "timezone": "UTC",
        "split": split,
        "profile_note": "Synthetic benchmark identity. Personal facts must come from source records.",
    }]

    event_specs = [
        ("role_acceptance", "2026-01-08", "accepted_role", role, "disclosed"),
        ("initial_goal", "2026-01-08", "career_goal", initial_goal, "disclosed"),
        ("work_preference", "2026-01-08", "work_preference", "remote", "disclosed"),
        ("true_start_date", correct_date, "job_start_date", correct_date, "disclosed_after_error"),
        ("temporary_state", "2026-05-18", "feels_exhausted", True, "disclosed"),
        ("goal_change", "2026-05-18", "career_goal", current_goal, "disclosed"),
        ("project_assignment", "2026-06-20", "leads_project", project, "disclosed"),
        ("mentor_assignment", "2026-06-20", "has_mentor", mentor, "disclosed"),
        ("office_location", "2026-07-22", "office_base", city, "disputed_in_sources"),
        ("project_review", milestone_date, "project_review_date", milestone_date, "disclosed"),
        ("relocation_status", "2026-11-14", "has_relocation_plan", False, "disclosed"),
        ("private_fact", "2026-01-01", "favorite_book", f"Private book {index}", "never_disclosed"),
    ]
    events = []
    for event_no, (event_type, valid_from, predicate, obj, disclosure) in enumerate(event_specs, 1):
        event_id = f"scaled_{user_id}_event_{event_no:03d}"
        events.append({
            "event_id": event_id,
            "benchmark_version": BENCHMARK_VERSION,
            "user_id": user_id,
            "event_type": event_type,
            "valid_from": valid_from,
            "valid_to": None,
            "facts": [{
                "fact_id": f"scaled_{user_id}_fact_{event_no:03d}",
                "subject_id": user_id,
                "predicate": predicate,
                "object": obj,
            }],
            "caused_by": [f"scaled_{user_id}_event_005"] if event_type == "goal_change" else [],
            "superseded_by": f"scaled_{user_id}_event_006" if event_type == "initial_goal" else None,
            "disclosure_status": disclosure,
        })

    claim_specs = [
        ("accepted_role", role, "asserted", "current", conv1, quotes["intro"], 1),
        ("career_goal", initial_goal, "asserted", "historical", conv1, quotes["intro"], 1),
        ("work_preference", "remote", "asserted", "current", conv1, quotes["intro"], 1),
        ("job_start_date", wrong_date, "reported_by_other", "superseded", email1, quotes["wrong_date"], 1),
        ("job_start_date", correct_date, "corrected", "current", conv3, quotes["correction"], 1),
        ("feels_exhausted", True, "asserted", "historical", conv2, quotes["shift"], 1),
        ("career_goal", current_goal, "asserted", "current", conv2, quotes["shift"], 1),
        ("leads_project", project, "reported_by_other", "current", email2, quotes["assignment"], 1),
        ("has_mentor", mentor, "reported_by_other", "current", email2, quotes["assignment"], 1),
        ("office_base", city, "reported_by_other", "disputed", chat2, quotes["office_one"], 1),
        ("office_base", possible_city, "reported_by_other", "disputed", chat2, quotes["office_two"], 2),
        ("has_relocation_plan", possible_city, "hypothetical", "excluded", cal1, quotes["move"], None),
        ("lives_in", possible_city, "reported_by_other", "excluded", chat1, quotes["wrong_person"], 1),
        ("project_review_date", milestone_date, "asserted", "confirmed", cal2, quotes["milestone"], None),
        ("work_preference", "remote", "asserted", "current", conv4, quotes["closing"], 1),
        ("project_deadline", uncertain_date, "uncertain", "candidate", conv4, quotes["closing"], 1),
    ]
    claims = []
    for claim_no, (predicate, obj, epistemic, status, source, quote, message_no) in enumerate(claim_specs, 1):
        subject_id = other_person.lower() if claim_no == 13 else user_id
        speaker_id = source["messages"][message_no - 1]["speaker_id"] if message_no else user_id
        claims.append({
            "claim_id": f"scaled_{user_id}_claim_{claim_no:03d}",
            "benchmark_version": BENCHMARK_VERSION,
            "user_id": user_id,
            "subject_id": subject_id,
            "speaker_id": speaker_id,
            "predicate": predicate,
            "object": obj,
            "polarity": "negative" if claim_no == 12 else "positive",
            "epistemic_status": epistemic,
            "memory_kind": "durative" if predicate in {"career_goal", "work_preference", "office_base", "has_mentor"} else "episodic",
            "status": status,
            "valid_from": source["created_at"],
            "valid_to": None,
            "time_precision": "day",
            "evidence": [_evidence(source, quote, message_no)],
            "review_status": "pending_human_review",
        })
    claim_times = {
        1: (f"{correct_date}T00:00:00+00:00", None),
        2: ("2026-01-08T09:00:00+00:00", "2026-05-17T23:59:59+00:00"),
        4: (f"{wrong_date}T00:00:00+00:00", None),
        5: (f"{correct_date}T00:00:00+00:00", None),
        6: ("2026-05-18T09:30:00+00:00", "2026-05-31T23:59:59+00:00"),
        14: (f"{milestone_date}T00:00:00+00:00", f"{milestone_date}T23:59:59+00:00"),
        16: (f"{uncertain_date}T00:00:00+00:00", None),
    }
    for claim_no, (valid_from, valid_to) in claim_times.items():
        claims[claim_no - 1]["valid_from"] = valid_from
        claims[claim_no - 1]["valid_to"] = valid_to

    def cid(number: int) -> str:
        return f"scaled_{user_id}_claim_{number:03d}"

    def eid(number: int) -> str:
        return f"scaled_{user_id}_event_{number:03d}"

    qa_specs: dict[str, list[tuple[str, str, list[str], list[dict[str, Any]], list[str], list[str], bool, str | None, str]]] = {
        "extraction": [
            (f"What role did {name} accept?", role, [role], [_evidence(conv1, quotes["intro"])], [cid(1)], [eid(1)], False, None, "single_source"),
            (f"Which company hired {name}?", company, [company], [_evidence(conv1, quotes["intro"])], [cid(1)], [eid(1)], False, None, "single_source"),
            (f"Which project will {name} lead?", project, [project], [_evidence(email2, quotes["assignment"])], [cid(8)], [eid(7)], False, None, "single_source"),
            (f"Who will mentor {name}?", mentor, [mentor], [_evidence(email2, quotes["assignment"])], [cid(9)], [eid(8)], False, None, "single_source"),
            (f"When is the {project} review?", milestone_date, [milestone_date], [_evidence(cal2, quotes["milestone"], None)], [cid(14)], [eid(10)], False, None, "structured_source"),
            (f"What work setting did {name} say suits focused work?", "Remote work", ["remote", "working remotely"], [_evidence(conv1, quotes["intro"])], [cid(3)], [eid(3)], False, None, "single_source"),
            (f"How did {name} describe their state after the launch?", "Exhausted", ["exhausted", "felt exhausted"], [_evidence(conv2, quotes["shift"])], [cid(6)], [eid(5)], False, None, "single_source"),
            (f"What possible move did {name}'s calendar mention?", f"A possible move to {possible_city}, explicitly not decided", [possible_city], [_evidence(cal1, quotes["move"], None)], [cid(12)], [eid(11)], False, None, "epistemic_status"),
            (f"Whose move to {possible_city} was mentioned in chat?", other_person, [other_person], [_evidence(chat1, quotes["wrong_person"])], [cid(13)], [], False, None, "speaker_attribution"),
            (f"What deadline did {name} say still needed confirmation?", uncertain_date, [uncertain_date], [_evidence(conv4, quotes["closing"])], [cid(16)], [], False, None, "uncertainty"),
        ],
        "temporal_reasoning": [
            (f"What was {name}'s career focus at the start of 2026?", initial_goal, [initial_goal], [_evidence(conv1, quotes["intro"])], [cid(2)], [eid(2)], False, None, "historical_state"),
            (f"What career goal was current for {name} after May?", current_goal, [current_goal], [_evidence(conv2, quotes["shift"]), _evidence(conv3, quotes["correction"])], [cid(7)], [eid(6)], False, None, "current_state"),
            (f"When did {name}'s stated career goal change?", "May 2026", ["2026-05", "May 2026"], [_evidence(conv2, quotes["shift"])], [cid(7)], [eid(6)], False, None, "date_normalization"),
            (f"Which goal came before the goal to {current_goal}?", initial_goal, [initial_goal], [_evidence(conv1, quotes["intro"]), _evidence(conv2, quotes["shift"])], [cid(2), cid(7)], [eid(2), eid(6)], False, None, "event_ordering"),
            (f"What start date was recorded before {name}'s correction?", wrong_date, [wrong_date], [_evidence(email1, quotes["wrong_date"])], [cid(4)], [], False, None, "superseded_state"),
            (f"What start date did {name} later correct the record to?", correct_date, [correct_date], [_evidence(conv3, quotes["correction"])], [cid(5)], [eid(4)], False, None, "correction_time"),
            (f"Did the launch exhaustion describe a permanent state?", "No. It was described as a state that month.", ["no", "temporary"], [_evidence(conv2, quotes["shift"])], [cid(6)], [eid(5)], False, None, "temporary_state"),
            (f"Which came first: the {project} assignment or its review?", "The project assignment came first", ["assignment", "project assignment"], [_evidence(email2, quotes["assignment"]), _evidence(cal2, quotes["milestone"], None)], [cid(8), cid(14)], [eid(7), eid(10)], False, None, "event_ordering"),
            (f"Was remote work mentioned only once in {name}'s history?", "No. It was mentioned in January and again in November.", ["no", "January and November"], [_evidence(conv1, quotes["intro"]), _evidence(conv4, quotes["closing"])], [cid(3), cid(15)], [eid(3)], False, None, "multi_session"),
            (f"As of March 2026, which start date was visible in the sources?", wrong_date, [wrong_date], [_evidence(email1, quotes["wrong_date"])], [cid(4)], [], False, None, "as_of_visibility"),
        ],
        "conflict_detection": [
            (f"Which start date should be treated as current for {name}?", correct_date, [correct_date], [_evidence(email1, quotes["wrong_date"]), _evidence(conv3, quotes["correction"])], [cid(4), cid(5)], [eid(4)], False, None, "explicit_correction"),
            (f"How are the two start-date claims related?", "The later statement explicitly corrects the earlier record", ["explicit correction"], [_evidence(email1, quotes["wrong_date"]), _evidence(conv3, quotes["correction"])], [cid(4), cid(5)], [eid(4)], False, None, "conflict_type"),
            (f"Should the earlier start-date record be erased?", "No. It should remain as superseded history.", ["no", "preserve it"], [_evidence(email1, quotes["wrong_date"]), _evidence(conv3, quotes["correction"])], [cid(4), cid(5)], [eid(4)], False, None, "preserve_history"),
            (f"What two office bases were reported for {name}?", f"{city} and {possible_city}", [f"{city} and {possible_city}"], [_evidence(chat2, quotes["office_one"]), _evidence(chat2, quotes["office_two"], 2)], [cid(10), cid(11)], [eid(9)], False, None, "source_disagreement"),
            (f"Is either reported office base authoritative?", "No. The sources disagree and one roster may be stale.", ["no", "disputed"], [_evidence(chat2, quotes["office_one"]), _evidence(chat2, quotes["office_two"], 2)], [cid(10), cid(11)], [eid(9)], False, None, "unresolved_dispute"),
            (f"Does {other_person}'s move conflict with {name}'s location?", "No. It concerns a different person.", ["no", "different person"], [_evidence(chat1, quotes["wrong_person"])], [cid(13)], [], False, None, "wrong_person_distractor"),
            (f"Are {name}'s two career goals a hard contradiction?", "No. They apply to different periods.", ["no", "temporal change"], [_evidence(conv1, quotes["intro"]), _evidence(conv2, quotes["shift"])], [cid(2), cid(7)], [eid(2), eid(6)], False, None, "temporal_change"),
            (f"Does the relocation calendar prove that {name} plans to move?", "No. It explicitly says no decision was made.", ["no", "not decided"], [_evidence(cal1, quotes["move"], None)], [cid(12)], [eid(11)], False, None, "hypothetical"),
            (f"Does the calendar title override its relocation note?", "No. The note limits it to a planning discussion.", ["no", "planning discussion"], [_evidence(cal1, quotes["move"], None)], [cid(12)], [eid(11)], False, None, "misleading_calendar_title"),
            (f"Why does the corrected start date outrank the onboarding sheet?", f"{name} directly corrected the earlier record to {correct_date}.", ["direct correction", correct_date], [_evidence(email1, quotes["wrong_date"]), _evidence(conv3, quotes["correction"])], [cid(4), cid(5)], [eid(4)], False, None, "evidence_authority"),
        ],
        "user_modeling": [
            (f"What work setting is consistently supported for {name}?", "Remote work", ["remote work", "remote"], [_evidence(conv1, quotes["intro"]), _evidence(conv4, quotes["closing"])], [cid(3), cid(15)], [eid(3)], False, None, "stable_preference"),
            (f"How did {name}'s career direction change during 2026?", f"It moved from {initial_goal} toward a goal to {current_goal}.", [initial_goal, current_goal], [_evidence(conv1, quotes["intro"]), _evidence(conv2, quotes["shift"])], [cid(2), cid(7)], [eid(2), eid(6)], False, None, "trajectory"),
            (f"What career goal is currently supported for {name}?", current_goal, [current_goal], [_evidence(conv2, quotes["shift"]), _evidence(conv3, quotes["correction"])], [cid(7)], [eid(6)], False, None, "current_state"),
            (f"Does the history establish that {name} is always exhausted?", "No. It supports a temporary post-launch state only.", ["no", "temporary"], [_evidence(conv2, quotes["shift"])], [cid(6)], [eid(5)], False, None, "trait_overreach"),
            (f"What ongoing work relationship is recorded for {name}?", f"{mentor} is mentoring {name} on {project}.", [mentor, "mentor"], [_evidence(email2, quotes["assignment"])], [cid(9)], [eid(8)], False, None, "relationship"),
            (f"What project focus is part of {name}'s current work?", project, [project], [_evidence(email2, quotes["assignment"]), _evidence(cal2, quotes["milestone"], None)], [cid(8), cid(14)], [eid(7), eid(10)], False, None, "current_role_context"),
            (f"Is relocation a settled part of {name}'s profile?", "No. A move was only discussed as a possibility.", ["no", "not settled"], [_evidence(cal1, quotes["move"], None), _evidence(conv4, quotes["closing"])], [cid(12)], [eid(11)], False, None, "hypothetical"),
            (f"Can one office base be added confidently to {name}'s profile?", "No. The two reports remain disputed.", ["no", "disputed"], [_evidence(chat2, quotes["office_one"]), _evidence(chat2, quotes["office_two"], 2)], [cid(10), cid(11)], [eid(9)], False, None, "counter_evidence"),
            (f"What repeated evidence supports {name}'s work preference?", "They said remote work supports focus in January and still suited them in November.", ["January and November", "remote"], [_evidence(conv1, quotes["intro"]), _evidence(conv4, quotes["closing"])], [cid(3), cid(15)], [eid(3)], False, None, "multi_session"),
            (f"Does the history establish a broad personality trait for {name}?", "No. It supports work preferences and states, not a broad personality label.", ["no", "not established"], [_evidence(conv1, quotes["intro"]), _evidence(conv2, quotes["shift"])], [cid(3), cid(6)], [], False, None, "unsupported_trait"),
        ],
        "abstention": [
            (f"Which university did {name} attend?", "The history does not establish that.", ["unknown", "not established"], [], [], [], True, "missing_fact", "missing_fact"),
            (f"What is {name}'s salary?", "The history does not establish that.", ["unknown", "not established"], [], [], [], True, "missing_fact", "missing_fact"),
            (f"Why does {name} dislike {manager}?", "The history does not establish that premise or cause.", ["not established", "insufficient evidence"], [], [], [], True, "unsupported_causal_claim", "unsupported_causal_claim"),
            (f"What is {name}'s authoritative office base?", "The history does not establish one; the reports conflict.", ["disputed", "not established"], [_evidence(chat2, quotes["office_one"]), _evidence(chat2, quotes["office_two"], 2)], [cid(10), cid(11)], [eid(9)], True, "unresolved_conflict", "unresolved_conflict"),
            (f"Has {name} decided to move to {possible_city}?", "No decision is established.", ["not decided", "unknown"], [_evidence(cal1, quotes["move"], None), _evidence(conv4, quotes["closing"])], [cid(12)], [eid(11)], True, "hypothetical_only", "hypothetical"),
            (f"Does {name} have a permanent health condition?", "The history does not establish that from one temporary state.", ["not established", "insufficient evidence"], [_evidence(conv2, quotes["shift"])], [cid(6)], [eid(5)], True, "trait_from_single_event", "insufficient_evidence"),
            (f"What is the confirmed project deadline?", "The only deadline mentioned is uncertain and unconfirmed.", ["unconfirmed", "unknown"], [_evidence(conv4, quotes["closing"])], [cid(16)], [], True, "uncertain_date", "uncertainty"),
            (f"Where is {name} moving, based on the chat message?", "The chat concerns another person, not the user.", ["wrong person", "not established"], [_evidence(chat1, quotes["wrong_person"])], [cid(13)], [], True, "wrong_person", "wrong_person_distractor"),
            (f"Which company will employ {name} next year?", "The history does not establish that future fact.", ["unknown", "not established"], [], [], [], True, "future_fact", "missing_fact"),
            (f"What is {name}'s relationship status?", "The history does not establish that.", ["unknown", "not established"], [], [], [], True, "missing_fact", "missing_fact"),
        ],
    }

    runtime_qa, gold_qa = [], []
    for capability in CAPABILITIES:
        for local_no, spec in enumerate(qa_specs[capability], 1):
            question, answer, acceptable, evidence, claim_ids, event_ids, abstain, reason, tag = spec
            case_id = f"scaled_{user_id}_qa_{capability}_{local_no:03d}"
            runtime = _common_case(case_id, user_id, split, "qa", capability, "multi_source" if len(evidence) > 1 else "focused")
            runtime["question"] = question
            runtime_qa.append(runtime)
            gold_qa.append({
                **runtime,
                "reference_answer": answer,
                "acceptable_answers": acceptable,
                "should_abstain": abstain,
                "abstention_reason": reason,
                "evidence": evidence,
                "required_claim_ids": claim_ids,
                "gold_event_ids": event_ids,
                "failure_tags": [tag],
                "review_status": "pending_human_review",
            })

    temporal_summary_specs = [
        ("Summarize how the user's career goal changed during 2026.", f"{name} began the year focused on {initial_goal}. In May, they said they wanted to {current_goal}. In September, they said that goal still held.", [eid(2), eid(6)], [cid(2), cid(7)], [_evidence(conv1, quotes["intro"]), _evidence(conv2, quotes["shift"]), _evidence(conv3, quotes["correction"])]),
        ("Summarize the start-date record and its later correction.", f"An onboarding sheet listed {wrong_date}. In September, {name} corrected it to {correct_date}. The older date remains in the history as a superseded value.", [eid(4)], [cid(4), cid(5)], [_evidence(email1, quotes["wrong_date"]), _evidence(conv3, quotes["correction"])]),
        ("Summarize the order of the project assignment and review.", f"{name} was assigned to lead {project} in June, and its review was scheduled for {milestone_date}.", [eid(7), eid(10)], [cid(8), cid(14)], [_evidence(email2, quotes["assignment"]), _evidence(cal2, quotes["milestone"], None)]),
    ]
    model_summary_specs = [
        ("Summarize the user's supported work preferences without overstating them.", f"{name} said remote work helped them focus in January and still suited them in November. That supports a work preference, not a broad personality trait.", [eid(3)], [cid(3), cid(15)], [_evidence(conv1, quotes["intro"]), _evidence(conv4, quotes["closing"])]),
        ("Summarize the user's current work context and unresolved details.", f"{name} is leading {project}, with {mentor} as mentor, and wants to {current_goal}. Two reports disagree about the office base, and no move has been decided.", [eid(6), eid(7), eid(8), eid(9), eid(11)], [cid(7), cid(8), cid(9), cid(10), cid(11), cid(12)], [_evidence(conv2, quotes["shift"]), _evidence(email2, quotes["assignment"]), _evidence(chat2, quotes["office_one"]), _evidence(chat2, quotes["office_two"], 2), _evidence(cal1, quotes["move"], None)]),
        ("Summarize what the history supports about the user's current and temporary states.", f"{name} currently wants to {current_goal}. They felt exhausted after one launch, but the history does not show that exhaustion as a lasting trait.", [eid(5), eid(6)], [cid(6), cid(7)], [_evidence(conv2, quotes["shift"]), _evidence(conv3, quotes["correction"])]),
    ]
    chosen = temporal_summary_specs + model_summary_specs[:2] if index % 2 else temporal_summary_specs[:2] + model_summary_specs
    runtime_summaries, gold_summaries = [], []
    temporal_no = model_no = 0
    for summary_no, spec in enumerate(chosen, 1):
        instruction, reference, event_ids, claim_ids, evidence = spec
        capability = "temporal_reasoning" if spec in temporal_summary_specs else "user_modeling"
        if capability == "temporal_reasoning":
            temporal_no += 1
            local_no = temporal_no
        else:
            model_no += 1
            local_no = model_no
        case_id = f"scaled_{user_id}_summary_{capability}_{local_no:03d}"
        runtime = _common_case(case_id, user_id, split, "summarization", capability, "longitudinal")
        runtime["instruction"] = instruction
        runtime_summaries.append(runtime)
        gold_summaries.append({
            **runtime,
            "reference_summary": reference,
            "should_abstain": False,
            "evidence": evidence,
            "gold_event_ids": event_ids,
            "required_claim_ids": claim_ids,
            "failure_tags": ["timeline" if capability == "temporal_reasoning" else "user_model"],
            "review_status": "pending_human_review",
        })

    pair_cycle = (
        ("extraction", "temporal_reasoning"),
        ("conflict_detection", "abstention"),
        ("user_modeling", "extraction"),
        ("temporal_reasoning", "conflict_detection"),
        ("abstention", "user_modeling"),
    )
    pair = pair_cycle[(index - 1) % 5]
    interactive_content = {
        "extraction": (f"Help me recall the project I was assigned and who is mentoring me.", f"The response should identify {project} and {mentor}, with source evidence.", [cid(8), cid(9)], [_evidence(email2, quotes["assignment"])], False, ["use_project_assignment", "cite_source_evidence"]),
        "temporal_reasoning": (f"What changed in my career direction this year?", f"The response should distinguish {initial_goal} from the later goal to {current_goal}.", [cid(2), cid(7)], [_evidence(conv1, quotes["intro"]), _evidence(conv2, quotes["shift"])], False, ["distinguish_historical_and_current", "cite_transition_evidence"]),
        "conflict_detection": (f"Which start date should I trust?", f"The response should use {correct_date}, explain the correction, and preserve {wrong_date} as superseded.", [cid(4), cid(5)], [_evidence(email1, quotes["wrong_date"]), _evidence(conv3, quotes["correction"])], False, ["apply_explicit_correction", "preserve_superseded_value"]),
        "abstention": (f"Tell me my confirmed office base.", "The response should say the two reports remain unresolved and avoid choosing one.", [cid(10), cid(11)], [_evidence(chat2, quotes["office_one"]), _evidence(chat2, quotes["office_two"], 2)], True, ["abstain_on_unresolved_conflict", "explain_conflicting_evidence"]),
        "user_modeling": (f"What work pattern can you safely infer about me?", "The response may identify the repeated remote-work preference but should not invent a personality trait.", [cid(3), cid(15)], [_evidence(conv1, quotes["intro"]), _evidence(conv4, quotes["closing"])], False, ["use_repeated_evidence", "avoid_personality_overreach"]),
    }
    runtime_interactive, gold_interactive = [], []
    for local_no, capability in enumerate(pair, 1):
        message, scenario, claim_ids, evidence, abstain, behaviours = interactive_content[capability]
        case_id = f"scaled_{user_id}_interactive_{capability}_{local_no:03d}"
        runtime = _common_case(case_id, user_id, split, "interactive", capability, "multi_turn")
        runtime.update({"scenario": scenario, "initial_user_message": message, "allowed_turns": 3})
        runtime_interactive.append(runtime)
        gold_interactive.append({
            **runtime,
            "should_abstain": abstain,
            "evidence": evidence,
            "required_claim_ids": claim_ids,
            "expected_behaviours": behaviours,
            "failure_tags": ["interactive_memory_use"],
            "review_status": "pending_human_review",
        })

    reviews: dict[str, list[dict[str, Any]]] = {name: [] for name in ("corrections", "conflicts", "abstentions", "timelines", "summaries", "interactive_expectations", "evidence")}
    reviews["corrections"].append(_review("corrections", user_id, "claim_pair", f"{cid(4)}__{cid(5)}", ["correction_language", "replacement_value", "preserved_history"]))
    reviews["conflicts"].append(_review("conflicts", user_id, "claim_pair", f"{cid(10)}__{cid(11)}", ["speaker_scope", "time_overlap", "resolution_status"]))
    reviews["timelines"].append(_review("timelines", user_id, "user_timeline", user_id, ["event_order", "valid_time", "disclosure_plan"]))
    for case in gold_qa:
        if case["should_abstain"]:
            reviews["abstentions"].append(_review("abstentions", user_id, "qa_case", case["case_id"], ["answerability", "reason", "rejected_evidence"]))
    for case in gold_summaries:
        reviews["summaries"].append(_review("summaries", user_id, "summary_case", case["case_id"], ["event_set", "time_state", "uncertainty", "reference_summary"]))
    for case in gold_interactive:
        reviews["interactive_expectations"].append(_review("interactive_expectations", user_id, "interactive_case", case["case_id"], ["expected_behaviours", "memory_use", "answerability"]))
    for target_type, records in (("claim", claims), ("qa_case", gold_qa), ("summary_case", gold_summaries), ("interactive_case", gold_interactive)):
        for item in records:
            target_id = item.get("claim_id", item.get("case_id"))
            checks = ["source_reference", "message_reference", "exact_quote", "user_scope", "as_of_cutoff"]
            if not item.get("evidence"):
                checks.append("empty_evidence_justified")
            reviews["evidence"].append(_review("evidence", user_id, target_type, target_id, checks))

    return {
        "users": users,
        "sources": sources,
        "events": events,
        "claims": claims,
        "runtime_qa": runtime_qa,
        "gold_qa": gold_qa,
        "runtime_summaries": runtime_summaries,
        "gold_summaries": gold_summaries,
        "runtime_interactive": runtime_interactive,
        "gold_interactive": gold_interactive,
        **reviews,
    }


def _schemas() -> dict[str, dict[str, Any]]:
    def field_schema(field: str) -> dict[str, Any]:
        if field in {"participants", "messages", "facts", "caused_by", "files", "checks", "failure_tags", "evidence", "required_claim_ids", "gold_event_ids", "acceptable_answers", "expected_behaviours"}:
            return {"type": "array"}
        if field in {"metadata", "counts", "capability_counts", "splits", "review", "output"}:
            return {"type": "object"}
        if field in {"allowed_turns", "records"}:
            return {"type": "integer"}
        if field in {"should_abstain", "frozen_for_tuning"}:
            return {"type": "boolean"}
        if field in {"valid_to", "superseded_by", "abstention_reason"}:
            return {"type": ["string", "null"]}
        if field == "object":
            return {}
        return {"type": "string"}

    def object_schema(name: str, required: list[str], optional: list[str] | None = None) -> dict[str, Any]:
        fields = required + (optional or [])
        return {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$id": f"https://longitudinal-memory.local/scaled-v1/{name}.schema.json",
            "type": "object",
            "required": required,
            "properties": {field: field_schema(field) for field in fields},
            "additionalProperties": False,
        }

    return {
        "user": object_schema("user", ["user_id", "display_name", "timezone", "split", "profile_note"]),
        "source": object_schema("source", ["source_id", "source_type", "user_id", "created_at", "ingested_at", "participants", "content", "messages", "metadata"]),
        "runtime_case": object_schema("runtime_case", ["case_id", "benchmark_version", "split", "user_id", "task", "capability", "as_of", "difficulty"], ["question", "instruction", "scenario", "initial_user_message", "allowed_turns"]),
        "oracle_event": object_schema("oracle_event", ["event_id", "benchmark_version", "user_id", "event_type", "valid_from", "valid_to", "facts", "caused_by", "superseded_by", "disclosure_status"]),
        "gold_claim": object_schema("gold_claim", ["claim_id", "benchmark_version", "user_id", "subject_id", "speaker_id", "predicate", "object", "polarity", "epistemic_status", "memory_kind", "status", "valid_from", "valid_to", "time_precision", "evidence", "review_status"]),
        "gold_case": object_schema("gold_case", ["case_id", "benchmark_version", "split", "user_id", "task", "capability", "as_of", "difficulty", "should_abstain", "evidence", "failure_tags", "review_status"], ["question", "instruction", "scenario", "initial_user_message", "allowed_turns", "reference_answer", "acceptable_answers", "abstention_reason", "reference_summary", "gold_event_ids", "required_claim_ids", "expected_behaviours"]),
        "prediction": object_schema("prediction", ["case_id", "benchmark_version", "user_id", "prediction_status", "output"]),
        "review_queue": object_schema("review_queue", ["review_id", "benchmark_version", "queue", "user_id", "target_type", "target_id", "status", "checks", "notes"]),
        "manifest": object_schema("manifest", ["manifest_version", "benchmark_version", "release_status", "dataset_sha256", "files", "counts", "capability_counts", "splits", "review"]),
    }


def main() -> None:
    manifest_path = Path("data/scaled-v1/manifest.json")
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("release_status") == "frozen":
            raise SystemExit("refusing to overwrite the frozen scaled_v1 release")

    combined: dict[str, list[dict[str, Any]]] = {}
    for index, values in enumerate(PEOPLE, 1):
        built = _build_user(index, values)
        for key, records in built.items():
            combined.setdefault(key, []).extend(records)

    files = [
        _record("data/scaled-v1/runtime/users.jsonl", "runtime", combined["users"]),
        _record("data/scaled-v1/runtime/sources.jsonl", "runtime", combined["sources"]),
        _record("data/scaled-v1/runtime/qa.jsonl", "runtime", combined["runtime_qa"]),
        _record("data/scaled-v1/runtime/summaries.jsonl", "runtime", combined["runtime_summaries"]),
        _record("data/scaled-v1/runtime/interactive.jsonl", "runtime", combined["runtime_interactive"]),
        _record("data/scaled-v1/oracle/events.jsonl", "oracle", combined["events"]),
        _record("data/scaled-v1/gold/claims.jsonl", "gold", combined["claims"]),
        _record("data/scaled-v1/gold/qa.jsonl", "gold", combined["gold_qa"]),
        _record("data/scaled-v1/gold/summaries.jsonl", "gold", combined["gold_summaries"]),
        _record("data/scaled-v1/gold/interactive.jsonl", "gold", combined["gold_interactive"]),
        _record("data/scaled-v1/predictions/predictions.jsonl", "prediction", []),
    ]
    for queue in ("corrections", "conflicts", "abstentions", "timelines", "summaries", "interactive_expectations", "evidence"):
        files.append(_record(f"data/scaled-v1/review_queues/{queue}.jsonl", "review", combined[queue]))
    for name, schema in _schemas().items():
        files.append(_json_file(f"schemas/scaled-v1/{name}.schema.json", "schema", schema))

    qa_counts = Counter(item["capability"] for item in combined["runtime_qa"])
    summary_counts = Counter(item["capability"] for item in combined["runtime_summaries"])
    interactive_counts = Counter(item["capability"] for item in combined["runtime_interactive"])
    review_counts = {queue: len(combined[queue]) for queue in ("corrections", "conflicts", "abstentions", "timelines", "summaries", "interactive_expectations", "evidence")}
    manifest = {
        "manifest_version": "1",
        "benchmark_version": BENCHMARK_VERSION,
        "release_status": "candidate",
        "dataset_sha256": _dataset_hash(files),
        "files": files,
        "counts": {
            "users": len(combined["users"]),
            "sources": len(combined["sources"]),
            "oracle_events": len(combined["events"]),
            "gold_claims": len(combined["claims"]),
            "qa": len(combined["runtime_qa"]),
            "summaries": len(combined["runtime_summaries"]),
            "interactive_scenarios": len(combined["runtime_interactive"]),
            "predictions": 0,
            "review_queue_items": review_counts,
        },
        "capability_counts": {
            "qa": {name: qa_counts[name] for name in CAPABILITIES},
            "summaries": {
                "temporal_reasoning": summary_counts["temporal_reasoning"],
                "user_modeling": summary_counts["user_modeling"],
            },
            "interactive": {name: interactive_counts[name] for name in CAPABILITIES},
        },
        "splits": {
            "strategy": "whole_user",
            "development": {
                "user_ids": ["user_001", "user_002"],
                "qa": 100,
                "summaries": 10,
                "interactive_scenarios": 4,
            },
            "test": {
                "user_ids": [f"user_{number:03d}" for number in range(3, 11)],
                "qa": 400,
                "summaries": 40,
                "interactive_scenarios": 16,
                "frozen_for_tuning": True,
            },
        },
        "review": {
            "automated_validation_status": "passed",
            "implementation_review_status": "complete",
            "human_review_status": "pending_sneha_review",
            "review_queue_status": "pending_human_review",
        },
    }
    Path("data/scaled-v1/manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()

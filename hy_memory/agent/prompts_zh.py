"""
Chinese prompt definitions.

When the user input is detected as Chinese, the prompts in this file replace the English ones,
so that the LLM answers in Chinese.
"""


# ================================================================
# EXTRACT PROMPT (Chinese)
# ================================================================
# System prompt (fixed, never passed through format)
EXTRACT_SYSTEM_PROMPT_ZH = """You are a memory extraction system (Memory Extractor).

Your job is to extract from the conversation:

* durable
* high-value
* useful for future conversations

user memories, and turn them into structured, retrievable, self-contained memories.

---

# Input

You will receive:

* Last k Messages
* New Messages
* Memory Date
* Current Date

How they are used:

* Last k Messages are only for:

  * coreference resolution
  * context understanding
* Memory Date is the only time anchor you may use to resolve time expressions
* Current Date is system metadata only and is not used for temporal reasoning

---

# Extraction principles

Extract only new information that appears for the first time in the NEW user messages.

Extract in priority order:

* long-term preferences
* interests and hobbies
* long-running projects
* future plans
* decisions
* constraints
* habits and routines
* values
* stable attitudes
* recurring activities
* significant experiences
* advice the user accepted

Do not extract:

* small talk
* pleasantries
* one-off emotions
* temporary context
* the assistant restating an old memory
* information already clearly present in the earlier context
* vague personality judgements

---

# Subject rules

Every memory must be self-contained, understandable out of context, written in third person, and have an explicit subject.
Which subject you use is decided by **who is speaking**:

| Speaker | Subject to write | Example |
| --- | --- | --- |
| `role: "user"` | the user | "The user ..." |
| Any speaker that carries a `name` and is not the user (e.g. `name: "John"`) | use that person's name | "John ..." |
| `role: "assistant"` with **no** `name` | the AI assistant | "The assistant ..." |

The two input sections identify speakers differently:

* **New Messages** is a JSON array -- read the `role` field and the optional `name` field.
* **Last k Messages** is rendered text -- the speaker is the bracketed label `[user]:` / `[John]:` / `[assistant]:`,`
  and the header may also carry a line such as "The user's name is Tim".

Rules:

* The user is the person these memories belong to. Even when you already know the user's name (from the `name` on
  their own messages, or from the name line in the header), still write "The user ..." and **never** "Tim ...".
* **An `role: "assistant"` message that carries a `name` is another real person, not the AI.**
  Use that person's name as the subject of their memories: "John ...".
* Only an `role: "assistant"` message without `name` is the AI assistant.
* Never invent a name for the AI assistant; it has none.

---

# Language rules

* The output language must match the input language
* Keep the original terms, brand names, formulas and code
* Do not transliterate or translate proper nouns

---

# Two kinds of memory

You extract two kinds of memory: **memory** and **intentions**.

## memory

Everything worth remembering long term goes here -- both about the user and about other named real people in the conversation:

* preferences, opinions, attitudes, values
* events, actions, experiences
* habits, behaviour patterns
* relationships, stable traits
* objective facts, decisions, constraints
* activities currently under way

Extract preferences the user expresses implicitly as well:

* voluntarily resuming an activity -> positive interest
* avoiding something long term -> negative preference
* repeated voluntary engagement -> sustained interest

Do not separate "what kind of person the user is" from "what happened". Both go into `memory`.

## intentions

Forward-looking items the user **intends or plans to do** later: actionable, and meaningless once they are done or expired.

For example:

* "The user has a job interview next Tuesday."
* "The user plans to move at the end of the month."
* "The user plans to finish the report before Friday."

Not intentions (these go into `memory`):

* stable preferences or traits ("The user likes basketball")
* completed past events ("The user moved to Beijing last year")
* vague goals with no concrete future action

### valid_until

Give every intention a `valid_until`: the date it stops being valid (deadline / target date),
resolving relative time against **Memory Date** (for example "next Tuesday" -> the matching absolute date).
Set it to `null` when the deadline cannot be determined.

---

# OWNER

Every memory and intention must carry an `owner` saying which memory store it belongs to. Only two values exist:

* `"user"` -- everything that lives in the user's own store. That covers facts / preferences / traits about the user themselves,
  **and also facts about other named real people in the conversation** -- those belong in the user's memories just as much.
* `"agent"` -- information supplied by the AI assistant (an `role: "assistant"` message without `name`),
  (recommendations, confirmations, material it looked up), or things the user wants the **assistant** to do later (for example "remind me in 8 hours").

A named real person is **never** `owner: "agent"` -- they are a person, not the AI, even when their message carries
`role: "assistant"`. Their memories are `owner: "user"` and their own name is the subject of the content.

Decide `owner` item by item. Default to `"user"` when it cannot be determined.

---

# Time rules

Use Memory Date to resolve:

* tomorrow
* next week
* last month
* recently

and other relative time expressions.

If it cannot be resolved safely:

* drop the relative time expression
* keep the memory free of time dependencies

Never invent a temporal order.

**"It just happened" is itself an event and must be kept.** When the user describes something as just finished ("I just got back", "I just finished it",
"this morning", "yesterday"), the completion time is the Memory Date. Write out **the event itself** with that date;
do not fold it into a background statement.

* "I just got back from a trip with my family"
  -> "The user returned from a family trip on <Memory Date>."
  Wrong -> "The user goes travelling with their family." (the return date is exactly the answer to "when did you get back"; lose it and that question can never be answered)

If the conversation gives the start and the end of something separately, record both; do not compress them into one time range.

---

# Quality requirements

Every memory must be:

* information complete
* self-contained
* suited to retrieval
* information dense

Keep the details that matter:

* people's names
* product names
* places
* numbers
* formulas
* constraints

Never generalise something specific into a category; write what was actually said.

* "Banff" -> keep "Banff", do not write "the Rocky Mountains"
* "a three-year-old Border Collie named Luna" -> keep the breed, the age and the name

**Details supplied in the other speaker's turn count just as much.** A fact is often built across turns: the other speaker states a specific name and
the user confirms it. The subject rules decide **whose** memory it is, but that never licenses dropping the detail.

* [Sam]: "Did you go skiing in Banff?"
  [user]: "Yes, the skiing was amazing!"
  -> "The user went skiing in Banff and thought the snow conditions were great."
  Wrong -> "The user likes winter sports." (the place and the activity were given by two different speakers, and both are established)

Avoid:

* fragmented memories that lack context
* pure emotional noise
* information with no long-term value

Cover independent future retrieval needs with as few memories as possible:

* Compression means merging, not omitting. Every distinct, durable proposition in the new messages appears exactly once.
* Merge details that share a subject and the same real event, state, preference, plan or decision into one memory; a complete event may take 1-3 sentences.
* Do not output a broad summary of one piece of evidence together with several narrow rewrites of it.
* Split only when the subject differs, the events happened independently, or the states differ.
* If splitting would lose the relationship, keep the cause, advice or consequence in the same memory as the fact it explains.
* Do not merge events that happened independently on different dates just because they share a topic.
* Keep the time precision of the source text. Unless the date is explicit or uniquely determinable, do not rewrite vague times such as "last weekend" or "last summer" into exact dates; "the weekend before the Memory Date" is fine.

---

# TAGS

Every memory must contain 1-3 tags.

Requirements:

* use lowercase for topic tags (travel, health, work)
* keep proper nouns capitalised (Jasper, Canada, Banff)
* keep them short
* keep the topic unambiguous

For example:

* ai
* work
* travel
* gaming
* research
"""

STRUCTURE_METADATA_PROMPT_ZH = """

---

# Structured organisation

Every `memory` must also output:

* `topic_paths`: a backward-compatible field holding zero to three semantic paths, broad to narrow.
* `event_facet`: contains only `subject`, `participants`, `action`, `objects`, `locations`, `state`;
  `state` may only be `planned`, `ongoing`, `completed` or `unknown`.
* `topic_assignments`: zero to six structural assignments separated by subject, each containing `subject`, `role`, `path`.
  A `path` consists of `{name, type}` segments, and `type` may only be `topic`, `facet`, `series`, `episode`:
  use topic for a broad area and facet for a specific activity or semantic facet; use series only when the same specific activity or state line occurs more than once;
  use episode only for a single real event with clear boundaries. A shared event may be assigned to each explicit participant separately, while the text still
  keeps exactly one canonical memory.
  The type order must be `topic -> facet -> series -> episode`; skipping levels is allowed, repeating or reversing them is not.

The Extractor only sees the current write and cannot prove repetition across occurrences, so it must not emit series segments; emit only
Topic/Facet/Episode here. Only Reconcile may create a Series, after comparing independent historical writes.

This is semantic organisation at write time. Never infer Series/Episode from path depth, and never invent assignments out of proper nouns, broad tags or unrelated
co-occurrence; write only the most specific level the evidence supports, and use empty values when unsure.
Never encode question-shaped forms such as when/how many/who, plurals or ordinals into these fields.
"""

EXTRACT_PROMPT_ZH = """## Last k Messages

{last_messages}

---

## New Messages

{content}

---

## Memory Date

{memory_at}

---

## Current Date

{current_date}

---


# Output format

Return a single JSON object only.

Format:

```json
{{
  "memory": [
    {{
      "content": "...",
      "tags": ["..."],
      "owner": "user"
    }}
  ],
  "intentions": [
    {{
      "content": "...",
      "tags": ["..."],
      "valid_until": "YYYY-MM-DD",
      "owner": "user"
    }}
  ],
  "basic_info": {{}}
}}
```

Rules:

* memory:
  Everything worth remembering long term -- preferences, opinions, attitudes, traits, events, plans, experiences, habits, decisions.
  It may be about `[user]` or about other named real people (`[John]`); for the subject to use see "# Subject rules" above

* intentions:
  Forward-looking items the user (or another named real person) intends or plans to do later, with `valid_until` (deadline / target date,
  resolved against Memory Date); set it to `null` when the deadline cannot be determined

* owner (required on every memory and intention): `"user"` or `"agent"`
  (see "# OWNER" above; default to `"user"` when it cannot be determined)

* basic_info: **the user's own profile only**.
  Fill a stable profile field (name, age, location, occupation, employer) only when the NEW user messages state it **explicitly**.
  Never infer and never invent. Output `{{}}` when no field is stated explicitly.
  * Exception for `name`: the `name` field on the user's own message, or a header line of the form
    "The user's name is X" are both authoritative sources for `name`
    -- even if the user never states their own name out loud.
  * **Never** put another real person's attributes in here. If John (an assistant-role message carrying a `name`) says
    "I work as a nurse in Berlin", that is a `memory` ("John is a nurse in Berlin."),
    **not** `occupation` / `location` in `basic_info`.

* Do not repeat information across sections

* tags must be 1-3 keywords (topics lowercase; proper nouns such as place and people names keep their capitalisation)

* Use [] or {{}} when there is no content

* Output valid JSON only

Now output the JSON content"""


EXTRACT_PROMPT_ZH_NO_INTENTION = """## Last k Messages

{last_messages}

---

## New Messages

{content}

---

## Memory Date

{memory_at}

---

## Current Date

{current_date}

---


# Output format

Return a single JSON object only.

Format:

```json
{{
  "memory": [
    {{
      "content": "...",
      "tags": ["..."],
      "owner": "user"
    }}
  ],
  "basic_info": {{}}
}}
```

Rules:

* memory:
  Everything worth remembering long term -- preferences, opinions, attitudes, traits, events, experiences, habits, decisions,
  **plus future plans and upcoming arrangements** (do not open a separate intentions bucket).
  It may be about `[user]` or about other named real people (`[John]`); for the subject to use see "# Subject rules" above

* owner (required on every memory): `"user"` or `"agent"`
  (see "# OWNER" above; default to `"user"` when it cannot be determined)

* basic_info: **the user's own profile only**.
  Fill a stable profile field (name, age, location, occupation, employer) only when the NEW user messages state it **explicitly**.
  Never infer and never invent. Output `{{}}` when no field is stated explicitly.
  * Exception for `name`: the `name` field on the user's own message, or a header line of the form
    "The user's name is X" are both authoritative sources for `name`
    -- even if the user never states their own name out loud.
  * **Never** put another real person's attributes in here.

* Do not repeat information across sections
* tags: 1-3 keywords; topics lowercase, proper nouns keep their capitalisation
* Do not output the `intentions` field

"""



# ================================================================
# EXTRACT FEW-SHOT examples (Chinese)
# ================================================================
# Appended to the end of the system prompt only when few_shot_enabled is on.
# Every example matches the real input structure (Last k Messages / New Messages / Memory Date)
# and the output schema (memory / intentions / basic_info, with owner and tags),
# and ends with one sentence of commentary on the key point.
EXTRACT_FEW_SHOT_ZH = """
---

# Examples

## Example 1 -- boundary: old messages are only for coreference resolution, never re-extracted

Last k Messages:
[{"role": "user", "content": "I have been visiting the stray dogs at the downtown shelter lately."},
 {"role": "assistant", "content": "That is great -- have any of them caught your eye?"}]
New Messages:
[{"role": "user", "content": "Yeah, I brought her home yesterday -- a three-year-old Border Collie named Luna."}]
Memory Date: 2025-04-10

Output:
```json
{
  "memory": [
    {"content": "The user adopted a three-year-old Border Collie named Luna from the downtown shelter on 2025-04-09.", "tags": ["pets", "family"], "owner": "user"}
  ],
  "intentions": [],
  "basic_info": {}
}
```
The old messages are only used to resolve "her" and to fill in context ("the downtown shelter"). Do not re-output old information such as "the user is considering adopting a dog"; extract only the genuinely new fact (the adoption is complete), and use Memory Date to resolve "yesterday" into an absolute date.

## Example 2 -- condensing a long agent reply

New Messages:
[{"role": "user", "content": "Write me a short essay arguing why remote work improves productivity."},
 {"role": "assistant", "content": "Sure -- opening: 'Remote work is reshaping how teams collaborate ...' [about 2000 words of body text omitted] ... closing: 'In short, more autonomy and fewer interruptions bring higher output.'"}]
Memory Date: 2025-03-22

Output:
```json
{
  "memory": [
    {"content": "The assistant wrote a short essay for the user arguing that remote work improves productivity because of stronger autonomy and fewer interruptions.", "tags": ["writing", "remote-work"], "owner": "agent"}
  ],
  "intentions": [],
  "basic_info": {}
}
```
The agent reply is long. Never store the whole essay verbatim; distil it into topic plus thesis. The owner is "agent" because the assistant produced the content. Bloated source text makes a memory unusable.

## Example 3 -- compound extraction: basic_info + intention + a summarised agent plan

New Messages:
[{"role": "user", "content": "Hi, I'm Kenji. I'm planning a trip to Japan next month - could you build me a 7-day itinerary?"},
 {"role": "assistant", "content": "# 7-day classic golden route in Japan (Tokyo -> Kyoto -> Nara -> Osaka)\\nFly into Tokyo, take the Shinkansen south to Kyoto, a day trip to Nara, and finally depart from Osaka Kansai airport.\\n\\n## Day 1: Arrive in Tokyo\\n- Land at Haneda/Narita and drop your bags at a hotel near Tokyo Station.\\n- Evening: Senso-ji, walk Nakamise shopping street, try taiyaki.\\n- Night: Shibuya Crossing + Shibuya Sky for sunset; izakaya dinner (skewers, sake).\\n\\n## Day 2: Tokyo markets and landmarks -- [omitted]\\n## Day 3: Shinkansen to Kyoto, Fushimi Inari, Gion -- [omitted]\\n## Day 4: Kyoto temples (Kiyomizu-dera, Kinkaku-ji, Ginkaku-ji) -- [omitted]\\n## Day 5: Nara day trip (feeding the deer in Nara Park, Todai-ji) -- [omitted]\\n## Day 6: Osaka (Osaka Castle, street food in Dotonbori) -- [omitted]\\n## Day 7: Osaka sightseeing, then depart from Kansai airport -- [omitted]\\n\\nTip: buy a 7-day JR Pass before you leave. Want me to adjust anything?"}]
Memory Date: 2025-09-15

Output:
```json
{
  "memory": [
    {"content": "The assistant built a 7-day classic golden-route Japan itinerary for the user: Tokyo (Asakusa, Shibuya), Shinkansen to Kyoto (Fushimi Inari, Kiyomizu-dera, Kinkaku-ji, Gion), a day trip to Nara (the deer park, Todai-ji), then Osaka (Osaka Castle, Dotonbori), departing from Kansai airport.", "tags": ["travel", "japan", "itinerary"], "owner": "agent"}
  ],
  "intentions": [
    {"content": "The user plans a 7-day trip to Japan in October 2025.", "tags": ["travel", "japan"], "valid_until": "2025-10-31", "owner": "user"}
  ],
  "basic_info": {"name": "Kenji"}
}
```
One turn produces three different buckets: the name goes into basic_info; the trip is a forward-looking intention with a deadline; and the assistant's long multi-section itinerary is summarised into **one** high-density agent memory -- keeping only the route skeleton and the key sights per city, **not** the day-by-day detail, the tips or the food lists. Storing the whole reply verbatim bloats the memory; the skeleton is what stays retrievable.

## Example 4 -- a later message shows the user adopted earlier advice, which is itself a fact

New Messages:
[{"role": "user", "content": "My 6-year-old refuses to eat any vegetables, is there anything I can do?"},
 {"role": "assistant", "content": "You can blend spinach or carrot into the pasta sauce -- you can barely see or taste it, and plenty of kids accept it that way."},
 {"role": "user", "content": "Good idea, I'll try it tonight. By the way, what bedtime is good at this age?"},
 {"role": "assistant", "content": "For a 6-year-old, falling asleep between 7 and 9 pm is recommended, giving roughly 10 to 11 hours of sleep."}]
Memory Date: 2025-06-20

Output:
```json
{
  "memory": [
    {"content": "The user has a 6-year-old who refuses to eat vegetables, and the user adopted the assistant's advice to blend spinach or carrot into the pasta sauce to hide them.", "tags": ["parenting", "food"], "owner": "user"},
    {"content": "The assistant advised the user that a 6-year-old should fall asleep between 7 and 9 pm and get about 10 to 11 hours of sleep.", "tags": ["parenting", "sleep"], "owner": "agent"}
  ],
  "intentions": [],
  "basic_info": {}
}
```
In the third message the user says "I'll try it tonight" -- an explicit adoption of the earlier advice. That adoption carries real value and must be recorded ("the user adopted ..."), not discarded as small talk. The sleep advice was just given and has not been adopted, so it is recorded as advice from the assistant (owner "agent").

## Example 5 -- RAG / injected context: never store injected text

New Messages:
[{"role": "user", "content": "[Retrieved material: the Eiffel Tower is 330 metres tall and was completed in 1889.] With that in mind, I plan to propose to my girlfriend on top of the Eiffel Tower this summer."}]
Memory Date: 2025-05-02

Output:
```json
{
  "memory": [],
  "intentions": [
    {"content": "The user plans to propose to their girlfriend on top of the Eiffel Tower in summer 2025.", "tags": ["relationship", "travel"], "valid_until": "2025-08-31", "owner": "user"}
  ],
  "basic_info": {}
}
```
The retrieved material in brackets (the tower's height, the year it was built) is RAG-injected reference material, not something the user said about themselves -- never store it. Extract only the user's own forward-looking plan. Text injected by the system or by RAG is context for the model and is never a memory.

## Example 6 -- the user shares reference material: extract the content, not the sharing act

New Messages:
[{"role": "user", "content": "Remember my grandma's dumpling recipe so it doesn't get lost. Filling: 500 g minced pork, 300 g napa cabbage, 2 tbsp light soy sauce, 1 tbsp sesame oil, a small knob of ginger. Salt the cabbage for 15 minutes, then squeeze out the water. Boil them, adding cold water twice, and they are ready when the dumplings float."},
 {"role": "assistant", "content": "Sure, noted!"}]
Memory Date: 2025-07-01

Output:
```json
{
  "memory": [
    {"content": "The user's grandma's dumpling recipe: the filling is 500 g minced pork, 300 g napa cabbage (salted for 15 minutes and then squeezed dry), 2 tbsp light soy sauce, 1 tbsp sesame oil and a small knob of ginger; boil them, adding cold water twice, done when the dumplings float.", "tags": ["food", "recipe", "family"], "owner": "user"}
  ],
  "intentions": [],
  "basic_info": {}
}
```
Extract the actual factual content (the ingredients, the quantities, the salting and squeezing step, the cooking method), not "the user shared a recipe" or "the user asked to remember a recipe". The sharing act has no value; the content is what is retrievable.

## Example 7 -- structured numbers and attributes kept in full

New Messages:
[{"role": "user", "content": "Note down my home-gym PRs for this block: squat 140kg x 3, bench 100kg x 5, deadlift 180kg x 2, overhead press 60kg x 6. Bodyweight 78kg, 3 sets per lift, 3 minutes rest between sets."},
 {"role": "assistant", "content": "Logged -- those are strong numbers!"}]
Memory Date: 2026-01-20

Output:
```json
{
  "memory": [
    {"content": "The user's current strength PRs are squat 140kg x 3, bench 100kg x 5, deadlift 180kg x 2 and overhead press 60kg x 6, at a bodyweight of 78kg.", "tags": ["fitness", "strength"], "owner": "user"},
    {"content": "The user's training plan is 3 sets per lift with 3 minutes of rest between sets.", "tags": ["fitness", "training"], "owner": "user"}
  ],
  "intentions": [],
  "basic_info": {}
}
```
Every number and unit (kg, reps, sets, rest time) is kept in full -- losing them would ruin the most retrievable information. It splits into two clean topics (the PRs and the training plan), but never rounds a value or drops one.

## Example 8 -- a long multi-topic conversation: do not extract only the first topic

New Messages:
[{"role": "user", "content": "This week was big -- I finally passed my PhD defence on Tuesday!"},
 {"role": "assistant", "content": "Huge congratulations! How does that feel?"},
 {"role": "user", "content": "Such a relief! I also signed a lease for an apartment in Berlin and I'm moving in next month. I've started learning German on Duolingo too -- a 30-day streak already. My partner Sophia is still in Madrid for now, and that part is hard."}]
Memory Date: 2025-11-08

Output:
```json
{
  "memory": [
    {"content": "The user passed their PhD defence on 2025-11-04.", "tags": ["education", "milestone"], "owner": "user"},
    {"content": "The user signed a lease for an apartment in Berlin.", "tags": ["housing"], "owner": "user"},
    {"content": "The user is learning German on Duolingo and has a 30-day streak.", "tags": ["language", "learning"], "owner": "user"},
    {"content": "The user's partner Sophia is staying in Madrid for now, which the user finds hard.", "tags": ["relationship"], "owner": "user"}
  ],
  "intentions": [
    {"content": "The user plans to move to Berlin next month (around December 2025).", "tags": ["housing", "relocation"], "valid_until": "2025-12-31", "owner": "user"}
  ],
  "basic_info": {}
}
```
Four or more topics -- the defence, the Berlin apartment, learning German, a partner in Madrid. Do not extract only the first topic. Note that one turn splits into completed facts (the signed lease) plus one intention (the upcoming move), and the feeling about Sophia carries emotion but also holds long-term relationship information, so it is kept.

## Example 9 -- two real people: a name on an assistant message means a real person

New Messages:
[{"role": "user", "name": "Tim", "content": "Just back from the climbing gym, third time this week -- I finally sent that overhanging route."},
 {"role": "assistant", "name": "John", "content": "Impressive! I switched to swimming recently; the shoulder I hurt last month still hurts."}]
Memory Date: 2025-08-14

Output:
```json
{
  "memory": [
    {"content": "The user climbs three times a week and recently completed a difficult overhanging route.", "tags": ["fitness", "climbing"], "owner": "user"},
    {"content": "John recently switched to swimming because the shoulder he injured around July 2025 still hurts.", "tags": ["fitness", "health"], "owner": "user"}
  ],
  "intentions": [],
  "basic_info": {"name": "Tim"}
}
```
John carries `role: "assistant"`, but he has a `name`, so he is another real person and not the AI: his memories use "John ..." as the subject and the `owner` stays `"user"` (they live in the user's store). The user's own memories still read "The user ..."; even though you know he is called Tim you must not write "Tim climbs three times a week". `basic_info.name` comes from the `name: "Tim"` on the user's own message -- it is an authoritative source even though Tim never states his name out loud.

## Example 10 -- counter-example: another person's attributes never go into basic_info

New Messages:
[{"role": "user", "name": "Tim", "content": "What have you been up to?"},
 {"role": "assistant", "name": "John", "content": "I'm a nurse at a hospital in Berlin, working night shifts. Pretty exhausting."}]
Memory Date: 2025-09-02

Output:
```json
{
  "memory": [
    {"content": "John is a nurse at a hospital in Berlin, works night shifts, and finds it exhausting.", "tags": ["work", "health"], "owner": "user"}
  ],
  "intentions": [],
  "basic_info": {"name": "Tim"}
}
```
The occupation and the location are **John's** attributes, so they go into `memory` with "John" as the subject and must **never** land in `basic_info` as `occupation` / `location` -- `basic_info` belongs to the user alone. Only `name` (Tim, from the `name` field on the user's message) belongs there. Note that the output language follows the input language (English in this example).

---
"""


# ================================================================
# SEARCH_QUERY PROMPT (Chinese)
# ================================================================

SEARCH_QUERY_PROMPT_ZH = """You are a search query generator. Given a set of newly extracted memories, generate a set of short search queries that will find related existing memories in the vector database.

The goal is to maximise recall -- find existing memories that are semantically related to the new ones even when the wording differs a lot.

## New memories:
{new_memories}

## Guidance

The search queries you generate should cover:
- the key themes, entities and topics mentioned in the memories
- paraphrased or more abstract versions of the core concepts
- related concepts that may exist in the user's memory store

Output a JSON array of strings (5-15 queries, short and focused):

["query1", "query2", "query3", ...]

Output only the JSON array, with no other text."""


# ================================================================
# RECONCILE PROMPT (Chinese)
# ================================================================


RECONCILE_PROMPT_ZH = """You are a memory reconciliation system (Memory Reconciler).

Your job is to reconcile new memories into the existing memory base while keeping:

* no information loss
* no contradictions
* high retrievability
* a compact structure
* no fragmentation

---

# Input

You will receive:

* Existing memories
* New memories
* Current date

---

# Existing memories

Each existing memory contains:

* memory_id
* content
* owner
* memory_at
* tags

---

# New memories

Each new memory contains:

* content
* owner
* tags

New memories have no ID.

---

# Operation types

---

## ADD

Consider the information:

* it is entirely new
* it does not conflict with any existing memory
* it does not belong to an existing topic

If all of that holds, use ADD.

Format:

```json
{{
  "op": "ADD",
  "content": "...",
  "owner": "user",
  "tags": ["..."]
}}
```

---

## UPDATE

Consider the case where:

* the new and the existing memory describe the same topic
* and there is no conflict between them
* and merging improves completeness or retrievability

If all of that holds, use UPDATE.

Requirements:

* merge without loss
* keep every piece of information that is still valid
* do not produce duplicate memories

### Supplementing the same event vs another occurrence

Use **UPDATE** only when the new memory supplements **the same real event** or the same durable fact.

If the same person has the same kind of activity or event again at another time (another trip, match, injury, purchase, health event, and so on),
keep it as a separate **ADD** with the date and details belonging to that occurrence. If an existing memory is explicitly another occurrence
of the same specific series, you may add `related: [{{"memory_id":"...","relation":"same_series"}}]`.

Do not merge events from several different dates into one very long count/timeline memory; the retrieval layer assembles the timeline from these linked L2 entries.

Format:

```json
{{
  "op": "UPDATE",
  "memory_id": "...",
  "content": "...",
  "owner": "user",
  "tags": ["..."]
}}
```

---

## SUPERSEDE

Use it only when:

* the new memory conflicts with the old one on the same dimension
* the two cannot both be true
* the old memory no longer represents the current state

For example:

* the user moves to another city
* the user changes job
* the user's relationship status changes
* a long-standing user preference clearly reverses

Do not use it for:

* extra detail
* wording differences
* synonymous phrasing
* preferences that can coexist

Requirements:

* content states only the new state
* do not copy the old information
* the system keeps the old node and attaches it to the evolution chain, so you need not handle historical versions

Format:

```json
{{
  "op": "SUPERSEDE",
  "memory_id": "...",
  "content": "...",
  "supersede_reason": "...",
  "owner": "user",
  "tags": ["..."]
}}
```

### Folding several old memories into one chain

If **two or more** existing memories describe the **same dimension** and the new memory is the latest change on that dimension,
you can fold them into one evolution chain. Use `memory_ids` (an ordered list, **oldest to newest**)
instead of a single `memory_id`:

```json
{{
  "op": "SUPERSEDE",
  "memory_ids": ["<oldest id>", "<newer id>"],
  "content": "<write only the latest state>",
  "supersede_reason": "...",
  "owner": "user",
  "tags": ["..."]
}}
```

The listed old memories are chained oldest to newest and the new memory becomes the head. `content` still describes only the **latest state** --
**never** copy the old content into it; the old content stays in the chain history and remains retrievable.

---

# Key rules

---

## No information loss

All information that is still valid must remain retrievable.

Do not lose:

* information in the new memories
* old information that was UPDATEd or SUPERSEDED

Note:

SUPERSEDE does not need to copy the old content, because the old node still exists.

---

## Duplicate detection

If a new memory is already fully covered by an existing memory:

Emit no operation at all.

---

## Avoid fragmentation

When several memories:

* cover the same topic
* hold compatible content

prefer merging them with UPDATE.

Independent events on different dates stay as separate ADDs; use `same_series` only when the subject and the specific activity or event type are the same.

---

## Granularity

Each memory should express only:

* one complete topic
* one clear state

Avoid:

* fragments that are too small
* entries that are too long
* several topics mixed together

---

## Subject rules

**Never merge or supersede across subjects.** The subject is the person the content describes: "The user ..." / "John ..." / "Mary ...".
Memories about different people are independent facts even when the sentence pattern is nearly the same -- "The user likes coffee" and "John likes coffee"
are two independent memories and must never UPDATE or SUPERSEDE each other.

---

## owner rules

`owner` (`user`/`agent`): copy it verbatim from the memory being written.
**Never merge or supersede across owners** -- a user fact and an assistant fact are two different memories even on the same topic.

---

## Time rules

`memory_at: null`

means the time is unknown.

Do not invent a temporal order.

Do not supersede something just because it "looks newer".

---

## NO FORK RULE

* SUPERSEDE either targets one existing memory (`memory_id`) or folds several memories on the **same dimension** into one chain
  (`memory_ids`, oldest to newest); UPDATE may only target one existing memory.
* The same existing memory must not be touched by more than one operation.
* Do not create branching supersede chains.

---

## Optional field: `related`

Any operation **may** carry a `related` array recording the relationship between the memory being written and some
**existing memory**. It does not change what is stored; it only records one extra link.

```json
{{
  "op": "ADD",
  "content": "...",
  "owner": "user",
  "tags": ["..."],
  "related": [{{"memory_id": "<id of an existing memory>", "relation": "causes"}}]
}}
```

Only these four relations are allowed:

* `causes` -- the two memories form a cause-and-effect pair (for example "started running every day" -> "knee started hurting")
* `elaborates` -- a refinement of an existing memory, or a later development of it
* `contrasts` -- the two are in tension but **both still hold** (if one makes the other stop holding,
  that is SUPERSEDE, not `contrasts`)
* `same_series` -- independent occurrences of the same subject doing the same specific activity or event type at different times
  (for example two different skiing trips)

Requirements:

* Link only to ids that appear in "Existing memories"; **never invent an id**.
* Emit it only when the relation is **clear and specific**. Two memories merely sharing a broad topic is **not** a relation,
  so do not emit `related` in that case.
* `same_series` may not rest on a broad topic such as health, travel, sport, food or work; it must be a concrete repeated action or event for the same subject.
* If there is nothing to link, omit the field entirely -- **zero links are the normal case**.
* Never use `related` in place of UPDATE / SUPERSEDE.

---

{few_shot_section}
###################
The input information follows!
###################


Current date: {current_date}

## Existing memories

{existing_memories}

## New Memories

{new_memories}

# Output format

Output a JSON array.

For example:

```json
[
  {{
    "op": "ADD",
    "content": "...",
    "owner": "user",
    "tags": ["..."]
  }},
  {{
    "op": "UPDATE",
    "memory_id": "...",
    "content": "...",
    "owner": "user",
    "tags": ["..."]
  }},
  {{
    "op": "SUPERSEDE",
    "memory_id": "...",
    "content": "...",
    "supersede_reason": "...",
    "owner": "user",
    "tags": ["..."]
  }}
]
```

If no change is needed:

```json
[]
```

Output valid JSON only.

Now output the JSON array."""

RECONCILE_STRUCTURE_METADATA_PROMPT_ZH = """

Additional structural output contract:

Every ADD, UPDATE or SUPERSEDE result must keep or fix `topic_paths`, `event_facet` and
`topic_assignments`.
`topic_paths` are semantic paths from broad to narrow, for example `["sport", "swimming"]`; never guess a parent-child relation from a shared broad
tag. `event_facet` contains only subject/participants/action/objects/locations/state.
`topic_assignments` gives typed topic/facet/series/episode paths per subject; Series/Episode must
be clearly supported by the facts themselves and must not be inferred from path depth; the type order must follow
`topic -> facet -> series -> episode` (levels may be skipped, never repeated or reversed). If Existing memories
already hold a path with the same subject, equivalent meaning
and compatible boundaries, reuse its exactly identical node names and types so that synonymous nodes do not fork. When an exact reuse is needed, output
`structure_references: [{"memory_id":"<existing_id>", "subject":"...",
"through":"topic|facet|series|episode"}]`; ids must come from Existing memories, and `through` means copying
that old path up to the given type, while a deeper suffix the new fact states explicitly stays a new node. If the subject, the specific activity or state line, or the
Episode boundary differs, forced reuse is forbidden. No structural field may encode
question-shaped forms such as when/how many/who, plurals or ordinals.

A Series may only be created at the reconcile stage: the new memory and at least one separate Existing memory must be different occurrences
of the same specific activity or state line. Propositions split out of the current write are not evidence of repeated occurrence.
"""


# Chinese reconcile few-shot examples (injected into {few_shot_section} only when few_shot_enabled is on).
# Each example gives existing/new input plus the expected ops and one sentence of commentary. Ids are abbreviated.
RECONCILE_FEW_SHOT_ZH = """
###################
Examples
###################

## Example 1 -- one new memory: one part conflicts with an existing chain, the other is unrelated

Existing memories:
[
  {"memory_id": "m1", "content": "The user works as a product manager at Stripe.", "owner": "user", "history_versions": [{"content": "The user worked as a product manager at a fintech startup."}]}
]
New memories:
[
  {"content": "The user has left Stripe and is now a product lead at Notion; they have also started taking pottery classes at weekends.", "owner": "user"}
]

Output:
```json
[
  {"op": "SUPERSEDE", "memory_id": "m1", "content": "The user is a product lead at Notion.", "supersede_reason": "employer changed from Stripe to Notion", "owner": "user", "tags": ["work"]},
  {"op": "ADD", "content": "The user has started taking pottery classes at weekends.", "owner": "user", "tags": ["hobby"]}
]
```
A **single** new memory contains two things: the job change (the same dimension as the existing employment chain m1, and m1 already carries history -> SUPERSEDE, becoming the new chain head) and an unrelated detail (pottery -> ADD). Split this one memory into different operations along its parts -- the new chain head describes only the employment state and must never carry the pottery.

## Example 2 -- folding two previously unchained old memories plus the new change into one 3-node chain

Existing memories:
[
  {"memory_id": "a1", "content": "The user is learning guitar.", "owner": "user", "memory_at": "2024-02-10 09:00"},
  {"memory_id": "a2", "content": "The user can now play a few complete songs on guitar.", "owner": "user", "memory_at": "2024-09-05 09:00"}
]
New memories:
[
  {"content": "The user now plays guitar at a local open mic night.", "owner": "user"}
]

Output:
```json
[
  {"op": "SUPERSEDE", "memory_ids": ["a1", "a2"], "content": "The user plays guitar at a local open mic night.", "supersede_reason": "guitar level evolving: learning -> playing songs -> performing on stage", "owner": "user", "tags": ["music", "hobby"]}
]
```
a1 and a2 had no chain relationship, yet they describe the same dimension (progress on guitar) and the new memory is the latest step. Folding them with `memory_ids` from oldest to newest makes one 3-node chain (a1 <- a2 <- the new head). The head `content` describes only the current state -- do not restate "learning" or "can play a few songs".

## Example 3 -- the new chain head must never copy the content of the old memory

Existing memories:
[
  {"memory_id": "c1", "content": "The user rents a one-bedroom apartment in Shibuya, Tokyo, for 180,000 yen a month, loves that the station is a 5 minute walk away, but complains that it is noisy at night and has no parking.", "owner": "user"}
]
New memories:
[
  {"content": "The user just bought and moved into a three-bedroom house in the suburbs of Yokohama, with a garden and a two-car garage.", "owner": "user"}
]

Output:
```json
[
  {"op": "SUPERSEDE", "memory_id": "c1", "content": "The user bought and moved into a three-bedroom house in the suburbs of Yokohama, with a garden and a two-car garage.", "supersede_reason": "home changed from a rented one-bedroom in Shibuya to an owned three-bedroom in Yokohama", "owner": "user", "tags": ["housing"]}
]
```
The new memory SUPERSEDEs c1 and becomes the chain head. The head `content` describes only the new house in Yokohama. Do **not** drag the vivid details of the old place (Shibuya, the 180,000 yen rent, the 5 minute walk to the station, the noise, the missing parking) into the new node -- they clearly belong to the old home, not the new one. They stay in the chain history and remain retrievable; copying them in would make the head self-contradictory (renting in Shibuya and owning in Yokohama at the same time).

## Example 4 -- a different subject means never merge, however similar the wording

Existing memories:
[
  {"memory_id": "d1", "content": "The user drinks black coffee every morning.", "owner": "user"}
]
New memories:
[
  {"content": "John drinks black coffee every morning.", "owner": "user"},
  {"content": "The user now drinks black coffee twice a day, once in the morning and once after lunch.", "owner": "user"}
]

Output:
```json
[
  {"op": "ADD", "content": "John drinks black coffee every morning.", "owner": "user", "tags": ["food"]},
  {"op": "UPDATE", "memory_id": "d1", "content": "The user drinks black coffee twice a day, once in the morning and once after lunch.", "owner": "user", "tags": ["food"]}
]
```
The two new memories are nearly identical in wording but have **different subjects**. "John ..." is about another person, so it can only be ADDed -- never UPDATE or SUPERSEDE d1, even though d1 says almost the same thing about the user. Only the memory whose subject is also the user may be merged with d1.

## Example 5 -- the same kind of event happens again: UPDATE into a count plus a timeline (do not ADD)

Existing memories:
[
  {"memory_id": "i1", "content": "John injured his ankle last season and needed time off plus physical therapy.", "owner": "user", "memory_at": "2023-05-21 19:48"}
]
New memories:
[
  {"content": "John twisted his ankle again during a recent game.", "owner": "user", "memory_at": "2023-10-13 13:50"}
]

Output:
```json
[
  {"op": "UPDATE", "memory_id": "i1", "content": "John has injured his ankle two times: first last season (around the 2022–2023 season), needing time off and physical therapy; second during a game around 2023-10-13.", "owner": "user", "tags": ["health", "basketball", "injury"]}
]
```
The same kind of event happened again -> UPDATE the old memory: state the **total number of times** and list each occurrence in time order. Do not ADD another "John injured his ankle", and do not SUPERSEDE (the first injury has not become false).
"""


# ================================================================
# SUMMARY PROMPT (Chinese)
# ================================================================

SUMMARY_PROMPT_ZH = """Write a concise summary of the conversation content below.

Content:
---
{content}
---

Memory date: {memory_date}
Current date: {current_date}

Requirements:
1. **Third person**: describe the user as "The user ..." -- do not use pronouns without an explicit antecedent.
2. **Length**: 1-3 sentences, at most 200 words.
3. **Priority (when the content exceeds the length limit)**:
   a) changes, decisions, commitments (highest)
   b) explicit preferences, attitudes, likes and dislikes
   c) key events and facts
   d) background information (lowest)
4. **Keep preference signals**: keep any like, dislike, attitude or opinion expressed directly or indirectly, even when it looks unimportant.
5. **Self-contained**: the reader should understand the summary without seeing the original conversation.
6. **Do not invent**: do not add information that is not present in the source content.
7. **Language**: the output language must match the input language (whatever language goes in, the same language comes out).
8. **Time handling**:
   - **Memory date** is when the conversation actually took place and is the only time anchor you may use to resolve relative time references in it.
   - **Current date** is today's system date (years may have passed since the memory date); do not use it to interpret what the user said.
   - If a memory date is provided, convert relative time expressions into absolute references against it ("last week" -> the corresponding date).
   - If the memory date is empty, rewrite the sentence so that it carries no time (do not leave raw expressions such as "last week" or "yesterday" in the output).

## Output contract

Strict formatting requirements:
1. Output only the summary text -- plain text of 1-3 sentences.
2. Do not wrap the output in quotes, backticks, code blocks or any markdown.
3. Do not add prefixes or labels (no "Summary:", no "Here is ...", and nothing of the sort).
4. Do not add trailing explanations or meta commentary.
5. If the content is too simple to be worth summarising, output one sentence describing the most salient element -- never an empty string.

Now generate the summary."""


ROLLING_SUMMARY_PROMPT_ZH = """You maintain the user's conversation summary for the current day, organised by topic.

You are given:
1. PREVIOUS SUMMARY -- the most recent summary for this user today (may be empty).
2. NEW TURNS -- the recent conversation, each entry starting with a line id [n].

Your task: group NEW TURNS by topic and, for each group, decide:
- If it is valuable and already has enough content -> write a summary for it. If the group continues the same topic as the PREVIOUS SUMMARY,
  rewrite an updated summary that merges the previous summary with the new turns ("is_continuation": true).
- If it is valuable but not yet enough to summarise -> put those line ids into "keep_line_ids" (held over for the next round).
- If it is not valuable and not worth recording -> omit those line ids entirely (drop them).

PREVIOUS SUMMARY:
---
{prev_summary}
---

NEW TURNS:
---
{turns}
---

Memory date: {memory_date}
Current date: {current_date}

Summary writing rules (identical to the standard summary):
1. Third person "The user ...". 2. 1-3 sentences each, at most 200 words. 3. Keep preferences, decisions and changes.
4. Self-contained. 5. Do not invent. 6. The output language must match the language of the conversation. 7. Resolve relative time against the memory date.

## Output contract -- strict JSON, no markdown or code blocks:
{{
  "summary_list": [
    {{"content": "<summary text>", "topic": "<short topic label>", "line_ids": [<int>, ...], "is_continuation": <true|false>}}
  ],
  "keep_line_ids": [<int>, ...]
}}

Output rules:
- Each line id appears in at most one place (in some summary's line_ids, or in keep_line_ids, or dropped and absent altogether);
  it never repeats across two groups.
- line_ids / keep_line_ids may only reference line ids that appear in NEW TURNS.
- If nothing is worth a summary yet, return {{"summary_list": [], "keep_line_ids": [...]}}.
Output JSON only."""

#!/usr/bin/env python

import sys
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

from dotenv import load_dotenv

# Resolve paths based on this file's location
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

# Make sure project root is on sys.path so we can import eval.*
sys.path.insert(0, str(PROJECT_ROOT))

# Load .env from project root (for OPENAI_API_KEY, etc.)
load_dotenv(PROJECT_ROOT / ".env")

from openai import OpenAI
from eval.judge_text import evaluate_text  # your existing evaluator


# ---------- Helper functions ----------

def extract_ideas(text: str) -> str:
    """Extract content between <ideas> and </ideas> tags."""
    start_tag = "<ideas>"
    end_tag = "</ideas>"

    start_idx = text.find(start_tag)
    if start_idx == -1:
        return ""

    start_idx += len(start_tag)
    end_idx = text.find(end_tag, start_idx)

    if end_idx == -1:
        return ""

    return text[start_idx:end_idx].strip()


def extract_block(text: str, tag: str) -> Optional[str]:
    """Extract content between <tag> and </tag>, return None if missing."""
    import re

    match = re.search(fr"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
    return match.group(1).strip() if match else None


def build_ideas_prompt(title: str, content: str) -> str:
    """Prompt to generate tangential/unrelated ideas for a given AITA post."""
    return f"""<Instructions>Your job is to write the most viral Reddit posts in the "am I the asshole" subreddit.

You will be given a title and content of an already popular post. Please generate a list of tangential, unrelated, potentially surprising, and interesting concepts or ideas that may match well with the existing post.

Respond with a list of the ideas in the following format:
<ideas>['idea1...', 'idea2...', ...]</ideas>.
</Instructions>

<Title>{title}</Title>

<Post>{content}</Post>
"""


def build_posts_prompt(title: str, content: str, ideas: str) -> str:
    """Prompt to generate two new AITA-style posts using given ideas."""
    return f"""<Instructions>Your job is to write the most viral Reddit posts in the "am I the asshole" subreddit.

You will be given a post's title, the post's original content, and a list of unrelated but interesting concepts/ideas. You need to generate two different posts, each between 300-800 words, by incorporating one idea per post from the list provided into the original post to create new posts that you think have the best chance to go viral. Each of the new posts should still have the same title as the original post.

IMPORTANT STYLE & CONTENT CONSTRAINTS:
- You MUST NOT copy any full sentences from the original <Post>.
- You MUST NOT reuse distinctive phrases or near-verbatim wording from the original <Post>.
- The new posts should be substantially different in wording, sentence structure, and narrative flow.
- You SHOULD preserve the core concept and moral dilemma of the original, but you are free to:
  - Change the setting, background, and backstory.
  - Change the pacing and structure of how events are revealed.
  - Add or modify specific scenes, examples, or conversations (as long as they stay plausible).
  - Adjust tone, emotional emphasis, and narrative voice while staying in first-person AITA style.
- Think of this as writing a new story inspired by the same situation and the new idea, not paraphrasing or lightly editing the original.

The resulting new stories must:
- Be plausible and realistic
- Be highly relatable
- Contain drama, conflict, and no easy answer
- Encourage a lot of comments and debate

For each of the two posts, you MUST also:
1. Explicitly state which single idea from the <ideas> list you chose to incorporate.
2. Explain briefly why you believe using that idea makes the post more likely to go viral (e.g., more moral ambiguity, higher stakes, more intense conflict, unusual twist, etc.).

Output format (VERY IMPORTANT):
- Put the full text of the two rewritten posts inside:
  - <NewPost1>...</NewPost1>
  - <NewPost2>...</NewPost2>

- For each post, ALSO output:
  - The idea used, copied exactly as it appears in the <ideas> list, inside:
    - <NewPost1Idea>...</NewPost1Idea>
    - <NewPost2Idea>...</NewPost2Idea>

  - A short explanation (2–5 sentences) of why that idea increases the post's viral potential, inside:
    - <NewPost1WhyViral>...</NewPost1WhyViral>
    - <NewPost2WhyViral>...</NewPost2WhyViral>

Do NOT include any extra commentary outside these tags. Do NOT output JSON. Only use the tags described above.</Instructions>

<Title>{title}</Title>

<Post>{content}</Post>

<ideas>{ideas}</ideas>
"""


def generate_new_posts(title: str, content: str, client: Optional[OpenAI] = None) -> Tuple[str, str]:
    """
    1. Generates ideas for the given title + content.
    2. Uses those ideas to generate two new AITA-style posts.
    3. Returns the two new posts as strings.
    """
    if client is None:
        client = OpenAI()

    # 1) Get ideas
    ideas_prompt = build_ideas_prompt(title, content)
    ideas_response = client.responses.create(
        model="gpt-5",
        input=ideas_prompt,
    )
    ideas_text = ideas_response.output_text
    ideas = extract_ideas(ideas_text)

    # 2) Use ideas to generate new posts
    posts_prompt = build_posts_prompt(title, content, ideas)
    posts_response = client.responses.create(
        model="gpt-5",
        input=posts_prompt,
    )
    out = posts_response.output_text

    new_post_1 = extract_block(out, "NewPost1")
    new_post_1_idea = extract_block(out, "NewPost1Idea")
    new_post_2 = extract_block(out, "NewPost2")
    new_post_2_idea = extract_block(out, "NewPost2Idea")

    if new_post_1 is None or new_post_2 is None:
        raise ValueError("Failed to extract one or both new posts from the model output.")

    return new_post_1, new_post_1_idea, new_post_2, new_post_2_idea


# ---------- Main public function for your loop ----------

def generate_and_save_for_file(
    title: str,
    content_file_path: str | Path,
    threshold: float = 0.8,
    client: Optional[OpenAI] = None,
) -> Dict[str, Any]:
    """
    High-level function you can call from other scripts/notebooks.

    Steps:
    - Read the content file.
    - Generate two new posts for (title, content).
    - Evaluate each post with evaluate_text.
    - If score >= threshold, save to generators/tony/<basename>_newpost{1,2}.txt
      with format:

        title: {title}

        {content}

    Returns a dict with scores and saved file paths, e.g.:

    {
        "post1": {"score": 0.83, "saved_path": Path(...) or None, "text": "..."},
        "post2": {"score": 0.79, "saved_path": None, "text": "..."},
    }
    """
    if client is None:
        client = OpenAI()

    content_path = Path(content_file_path)
    content_text = content_path.read_text(encoding="utf-8")
    base_name = content_path.stem

    # 1) Generate new posts
    post1, post1_idea, post2, post2_idea = generate_new_posts(title, content_text, client=client)

    # 2) Evaluate
    eval1 = evaluate_text(post1)
    score1 = eval1.get("score", 0.0)

    eval2 = evaluate_text(post2)
    score2 = eval2.get("score", 0.0)

    # 3) Prepare output dir
    output_dir = SCRIPT_DIR / "tony"
    output_dir.mkdir(parents=True, exist_ok=True)

    result: Dict[str, Any] = {
        "post1": {"score": score1, "saved_path": None, "text": post1},
        "post2": {"score": score2, "saved_path": None, "text": post2},
    }

    # 4) Save posts that pass threshold
    # if score1 >= threshold:
    #     out_path1 = output_dir / f"{base_name}_newpost1.txt"
    #     out_path1.write_text(f"title: {title}\n\n{post1}", encoding="utf-8")
    #     result["post1"]["saved_path"] = out_path1

    # if score2 >= threshold:
    #     out_path2 = output_dir / f"{base_name}_newpost2.txt"
    #     out_path2.write_text(f"title: {title}\n\n{post2}", encoding="utf-8")
    #     result["post2"]["saved_path"] = out_path2

    out_path1 = output_dir / f"{base_name}_newpost1_{str(score1)}.txt"
    out_path1.write_text(f"title: {title}\n\n{post1}\n\nIdea:{post1_idea}", encoding="utf-8")
    result["post1"]["saved_path"] = out_path1

    out_path2 = output_dir / f"{base_name}_newpost2_{str(score2)}.txt"
    out_path2.write_text(f"title: {title}\n\n{post2}\n\nIdea:{post2_idea}", encoding="utf-8")
    result["post2"]["saved_path"] = out_path2

    return result

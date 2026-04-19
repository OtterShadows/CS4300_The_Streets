"""
LLM chat route — only loaded when USE_LLM = True in routes.py.
Adds a POST /chat endpoint that performs LLM-driven RAG.

Setup:
  1. Add SPARK_API_KEY=your_key to .env
  2. Set USE_LLM = True in routes.py
"""
import json
import os
import re
import logging
from flask import request, jsonify, Response, stream_with_context
from infosci_spark_client import LLMClient

logger = logging.getLogger(__name__)


def llm_search_decision(client, user_message):
    """Ask the LLM whether to search the DB and which word to use."""
    messages = [
        {
            "role": "system",
            "content": (
                "You have access to a database of Keeping Up with the Kardashians episode titles, "
                "descriptions, and IMDB ratings. Search is by a single word in the episode title. "
                "Reply with exactly: YES followed by one space and ONE word to search (e.g. YES wedding), "
                "or NO if the question does not need episode data."
            ),
        },
        {"role": "user", "content": user_message},
    ]
    response = client.chat(messages)
    content = (response.get("content") or "").strip().upper()
    logger.info(f"LLM search decision: {content}")
    if re.search(r"\bNO\b", content) and not re.search(r"\bYES\b", content):
        return False, None
    yes_match = re.search(r"\bYES\s+(\w+)", content)
    if yes_match:
        return True, yes_match.group(1).lower()
    if re.search(r"\bYES\b", content):
        return True, "Kardashian"
    return False, None

# Use the LLM to modify the query to hopefully yield better IR system results.
# Input: LLM client, user query
# Output: 
#   - return_character: (TRUE if the IR should display a character for the results, FALSE o/w)
#   - content: modified query
def llm_modify_query(client, user_message):
    messages = [
        {
            "role": "system",
            "content": (
                "You are an assistant to help rewrite user queries to improve search results "
                "from an information retrieval system. "
                ""
                "Background:"
                "The IR system uses comments from subreddit r/Piratefolk, a forum in which fans of "
                "mange/anime series One Piece make jokes and discuss the plot, writing, and most "
                "importantly for our purposes, the characters. They often push certain 'agendas' for "
                "the characters they support or dislike. For example, they may comment, 'Luffy is the GOAT' "
                "or 'Luffy low diffs Arlong."
                "The IR system calculates all of the comments belonging to (i.e. mentioning) each character, "
                "and given a user query, compares the similarity of the query to the aggregated comments of "
                "each character, return the most similar character, and return the most similar comments to the query "
                "that mention that character. "
                ""
                "You must modify the given user query to one that would more accurately match relevant comments of r/Piratefolk. "
                "Example 1: 'Who is the most liked' character should be transformed to something like "
                "'is the GOAT. is the best character. i like. low diffs. neg diffs. carries.', etc. with similar comments "
                "that would match with how people talk in discussions of that character."
                ""
                "Example 2: 'Most bum character' -> 'bum. useless. overrated.' etc. "
                ""
                "Example 3: 'Who is the goat of wano?' -> 'wano country arc. goat. carries. carried. strongest. powerful.' "
                ""
                "Always include lots of synonyms (also including slang) in order to cast a wide net for retrieving comments."
                ""
                "IMPORTANT: If the user query seems like it is seeking some information for which a character would "
                "be a proper answer, like the examples above, begin your response with YES followed by one space and the "
                "modified query. If instead returning a character wouldn't make sense for the query, (e.g query is "
                "'luffy katakuri fight', where the user probably just wants to see comments discussing the fight) "
                "return NO_CHARACTER followed by one space and the modified query."
            )
        },
        {"role": "user", "content": user_message},
    ]
    print("Calculating LLM response for query modification...")
    response = client.chat(messages)
    print("Finished calculating LLM response for query modification...")
    content = (response.get("content") or "").strip().upper()
    print(f"Content: {content}")
    logger.info(f"LLM search decision: {content}")
    if re.search(r"\bNO_CHARACTER\b", content):
        return_character = False
    else:
        return_character = True
    return return_character, content
 




def register_chat_route(app, json_search):
    """Register the /chat SSE endpoint. Called from routes.py."""

    @app.route("/chat", methods=["GET"])
    def chat():
        print(f"\033[95mChat route enabled\033[0m")
        user_message = request.args.get("q", "").strip()
        if not user_message:
            return jsonify({"error": "Message is required"}), 400

        api_key = os.getenv("SPARK_API_KEY")
        if not api_key:
            return jsonify({"error": "SPARK_API_KEY not set — add it to your .env file"}), 500

        client = LLMClient(api_key=api_key)
        return_character, modified_query = llm_modify_query(client, user_message)
        print(f"Return character?: {return_character}")
        print(f"Modified query: {modified_query}\n")
        # return_character: TRUE if a character should be displayed for the results 
        character_and_comments_json = json.loads(json_search(modified_query))
        return character_and_comments_json

        # TODO: Code below is from the template. For generating a natural language answer for the user, I believe.
        # This would be Maureen's task to adapt for our project.
        # Note for later: this code below was assuming the "methods" for this route was "POST" not "GET"...
        # I'm not entirely sure how this affects this code, but it's likely whoever implements
        # the chat summarizing will have to create a new route (with method POST) specifically for
        # the LLM interpreting the character/comments that the IR system returned

        # if use_search:
        #     context_text = "\n\n---\n\n".join(
        #         f"Title: {ep['title']}\nDescription: {ep['descr']}\nIMDB Rating: {ep['imdb_rating']}"
        #         for ep in episodes
        #     ) or "No matching episodes found."
        #     messages = [
        #         {"role": "system", "content": "Answer questions about Keeping Up with the Kardashians using only the episode information provided."},
        #         {"role": "user", "content": f"Episode information:\n\n{context_text}\n\nUser question: {user_message}"},
        #     ]
        # else:
        #     messages = [
        #         {"role": "system", "content": "You are a helpful assistant for Keeping Up with the Kardashians questions."},
        #         {"role": "user", "content": user_message},
        #     ]

        # def generate():
        #     if use_search and search_term:
        #         yield f"data: {json.dumps({'search_term': modified_query})}\n\n"
        #     try:
        #         for chunk in client.chat(messages, stream=True):
        #             if chunk.get("content"):
        #                 yield f"data: {json.dumps({'content': chunk['content']})}\n\n"
        #     except Exception as e:
        #         logger.error(f"Streaming error: {e}")
        #         yield f"data: {json.dumps({'error': 'Streaming error occurred'})}\n\n"

        # return Response(
        #     # Stream the response to the client ("stream_with_context" is from Flask)
        #     stream_with_context(generate()),
        #     mimetype="text/event-stream",
        #     # Set this to prevent the browser from caching the response
        #     headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        # )

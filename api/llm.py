import re
import os
from openai import OpenAI

class LLMHandler:
    def __init__(self, model):
        self.model = model
        self._client = None

    @property
    def client(self):
        if self._client is None:
            self._client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        return self._client

    def find_best_match(self, query, search_results):
        prompt = f"""Query: {query}

Results:
{self._format_results(search_results)}

Based on the query and the search results, which result is the best match? 
Return your response in the following format:
Best match: [index of the best match]
Explanation: [Your explanation here]

If you can't determine a best match, explain why and suggest how to improve the query."""

        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant designed to find the best match for a given query among search results."},
                {"role": "user", "content": prompt}
            ]
        )

        content = response.choices[0].message.content
        return self._parse_response(content, search_results)


    def _format_results(self, search_results):
        formatted_results = ""
        for i, (doc, metadata) in enumerate(zip(search_results['documents'][0], search_results['metadatas'][0])):
            formatted_results += f"{i+1}. Document: {doc}\nMetadata: {metadata}\n\n"
        return formatted_results

    def _parse_response(self, content, search_results):
        match = re.search(r'Best match: (\d+)', content)
        if match:
            index = int(match.group(1)) - 1  # Convert to 0-based index
            if 0 <= index < len(search_results['metadatas'][0]):
                best_match = search_results['metadatas'][0][index]
                return best_match['video_id'], best_match['start_time'], content
        
        # If no valid match found, return None values with the explanation
        return None, None, content
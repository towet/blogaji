import os
import requests
import json
from crewai import Agent, Task, Crew
from langchain.tools import DuckDuckGoSearchRun
from langchain.llms.base import LLM
from typing import Optional, List
import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

# Set up the DeepSeek API key and URL
# Set up the DeepSeek API key and URL
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')
DEEPSEEK_API_URL = 'https://api.deepseek.com/chat/completions'
BLOG_POSTS_FILE = 'blog_posts.json'

def ask_deepseek(question):
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {DEEPSEEK_API_KEY}'
    }

    data = {
        'model': 'deepseek-chat',
        'messages': [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": question}
        ],
        'stream': False
    }

    response = requests.post(DEEPSEEK_API_URL, headers=headers, json=data,timeout=60)

    if response.status_code == 200:
        response_data = response.json()
        return response_data['choices'][0]['message']['content']
    else:
        return f"Error: {response.status_code}"

# Custom LLM class for DeepSeek
class DeepSeekLLM(LLM):
    temperature: float = 0.7

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        response = ask_deepseek(prompt)
        if stop:
            for s in stop:
                if s in response:
                    response = response.split(s)[0]
                    break
        return response

    @property
    def _llm_type(self) -> str:
        return "deepseek"

# Initialize the DeepSeek model
llm = DeepSeekLLM(temperature=0.7)

# Unsplash Image Tool
class UnsplashImageTool:
    name: str = "Unsplash Image Tool"
    description: str = "Fetches an image URL from Unsplash based on a search query."

    def run(self, query: str) -> str:
        access_key = "O9Nhgkzu7SOmZJsC_Nn4MAgKkt4JLPUlbeoVSaHqokM"
        search_url = "https://api.unsplash.com/search/photos"
        params = {
            "query": query,
            "per_page": 1,
            "client_id": access_key
        }

        try:
            response = requests.get(search_url, params=params)
            response.raise_for_status()
            search_results = response.json()

            if search_results["results"]:
                return search_results["results"][0]["urls"]["regular"]
            else:
                return "No image found for the given query."
        except requests.RequestException as e:
            return f"An error occurred: {e}"

# Instantiate the Unsplash Image Tool
unsplash_image_tool = UnsplashImageTool()

# Define agents
researcher = Agent(
    role='AI Researcher',
    goal='Find the latest and most impactful AI developments',
    backstory='You are an AI enthusiast with a keen eye for groundbreaking developments in the field. You stay updated with the latest trends and breakthroughs.',
    llm=llm,
    verbose=True
)

writer = Agent(
    role='Content Writer',
    goal='Create engaging and informative content about AI',
    backstory='You are a talented writer with a knack for explaining complex AI concepts in an accessible way. You enjoy crafting compelling narratives that captivate your audience.',
    llm=llm,
    verbose=True
)

editor = Agent(
    role='Content Editor',
    goal='Ensure the content is high-quality, engaging, and error-free',
    backstory='You are a meticulous editor with years of experience in polishing technical content. You have a sharp eye for detail and a deep understanding of effective communication.',
    llm=llm,
    verbose=True
)

formatter = Agent(
    role='Content Formatter',
    goal='Format and style the content for optimal readability and visual appeal',
    backstory='You are an expert in HTML and CSS, with a keen eye for design and user experience. You understand the importance of presentation and aesthetics in enhancing content.dont put the tittle of the introduction just write the introduction without the introduction tittle',
    llm=llm,
    verbose=True
)

publisher = Agent(
    role='Content Publisher',
    goal='Prepare the formatted content for web publication',
    backstory='You are responsible for preparing the content for the website, ensuring it fits the required format. You have a thorough understanding of the publishing process and attention to detail.do not leave out parts or over summerise ',
    llm=llm,
    verbose=True
)

# Define tasks
research_task = Task(
    description="Research the latest AI developments and identify a groundbreaking topic for a blog post. Provide a summary of the developments and why the chosen topic is significant.",
    agent=researcher,
    expected_output="A summary of the latest AI developments and a chosen topic for the blog post, highlighting its significance."
)

writing_task = Task(
    description="Write a 1000-word blog post about the chosen AI topic. Ensure it is engaging and informative for a general audience. Start with a catchy title and include a compelling introduction, informative body, and a strong conclusion.",
    agent=writer,
    expected_output="A 1000-word blog post with a catchy title, engaging introduction, informative body, and strong conclusion, covering the chosen AI topic."
)

editing_task = Task(
    description="Review and edit the blog post for clarity, coherence, and correctness. Ensure it is engaging, free of errors, and flows well. Maintain the title on its own line at the beginning.",
    agent=editor,
    expected_output="An edited and polished version of the blog post, free of errors and with improved clarity and coherence."
)

formatting_task = Task(
    description="Format the blog post using HTML tags. Use appropriate headings (h1 for title, h2 for subtitles), paragraphs, bullet points where necessary, and add inline styles for improved readability. Ensure the content is visually appealing and well-organized. don not osummerise ensure the whole content required is in the file after the tittle display the teaser without the word teaser and then the rest of the content can follo ",
    agent=formatter,
    expected_output="An HTML-formatted version of the blog post with appropriate styling and structure.with all the contents needed"
)

publishing_task = Task(
    description="Prepare the formatted blog post for web publication. Create a 60-word teaser for the blog preview, and ensure all elements (title, content, teaser) are properly separated.",
    agent=publisher,
    expected_output="A fully formatted blog post ready for web publication, including a separate title, HTML-formatted content, and a 60-word teaser."
)

# Create the crew
content_crew = Crew(
    agents=[researcher, writer, editor, formatter, publisher],
    tasks=[research_task, writing_task, editing_task, formatting_task, publishing_task],
    verbose=2
)

def generate_blog_post():
    try:
        result = content_crew.kickoff()
        print(f"Type of result: {type(result)}")
        print(f"Content of result: {result}")

        # Check if result is a CrewOutput object
        if hasattr(result, 'result'):
            content = result.result
        else:
            content = str(result)

        # Parse the result to extract title, content, and teaser
        # Assuming the formatter has wrapped the title in <h1> tags
        title_start = content.index('<h1>') + 4
        title_end = content.index('</h1>')
        title = content[title_start:title_end].strip()

        # Extract the full content (everything after the title)
        full_content = content[title_end + 5:].strip()

        # Extract the teaser (assuming it's the first paragraph after the title)
        teaser_end = full_content.index('</p>') + 4
        teaser = full_content[:teaser_end].strip()

        # Remove HTML tags from the teaser
        teaser = teaser.replace('<p>', '').replace('</p>', '')

        # Use the Unsplash Image Tool to get the image URL based on the title
        image_url = unsplash_image_tool.run(title)

        # Generate a timestamp for the filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

        # Create the blog post data
        new_post = {
            'id': timestamp,
            'title': title,
            'content': full_content,
            'teaser': teaser,
            'date': datetime.datetime.now().isoformat(),
            'image': image_url
        }

        # Save to JSON file
        if os.path.exists(BLOG_POSTS_FILE):
            with open(BLOG_POSTS_FILE, 'r') as file:
                posts = json.load(file)
        else:
            posts = []

        posts.append(new_post)

        with open(BLOG_POSTS_FILE, 'w') as file:
            json.dump(posts, file, indent=2)

        logging.info(f"Blog post data has been saved to {BLOG_POSTS_FILE}")
        return new_post

    except Exception as e:
        logging.error(f"Error during task execution: {e}")
        return None

if __name__ == "__main__":
    new_post = generate_blog_post()
    if new_post:
        print(f"New blog post generated: {new_post['title']}")
    else:
        print("Failed to generate new blog post.")

import os
import requests
import json
from crewai import Agent, Task, Crew
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.llms.base import LLM
from typing import Optional, List
import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

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

    response = requests.post(DEEPSEEK_API_URL, headers=headers, json=data)

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

# Define agents with refined prompts
researcher = Agent(
    role='Technology Research Specialist',
    goal='Uncover groundbreaking Technological developments ',
    backstory='You\'re a renowned Technology researcher with a Ph.D. in Computer Science and years of experience at top tech companies. Your expertise allows you to spot emerging trends and understand their implications.',
    llm=llm,
    verbose=True
)

writer = Agent(
    role=' Technology Storyteller',
    goal='Craft an engaging narrative that makes complex Technological concepts accessible and exciting to a broad audience',
    backstory='You\'re a best-selling technology author with a knack for turning intricate topics into page-turners. Your writing has inspired countless readers to explore the world of Technology.',
    llm=llm,
    verbose=True
)

editor = Agent(
    role='Tech Content Perfectionist',
    goal='Elevate the blog post to professional publishing standards while maintaining its engaging essence',
    backstory='You\'ve edited for top tech publications and have a deep understanding of tecnological concepts and effective communication strategies.',
    llm=llm,
    verbose=True
)

formatter = Agent(
    role='Digital Content Architect',
    goal='Create a visually stunning and user-friendly blog post layout based on the provided content do not add your own content ensure all the contents are captured',
    backstory='You\'re a web design expert who specializes in creating immersive digital reading experiences. Your designs consistently receive praise for their aesthetics and usability.while formatting do not include the introduction tittle  just proceed to write the introduction  ',
    llm=llm,
    verbose=True
)

publisher = Agent(
    role='Content Marketing Strategist',
    goal='Maximize the blog post\'s reach and impact across digital platforms',
    backstory='You\'ve led content strategies for major tech brands, consistently increasing engagement and conversion rates through data-driven approaches.',
    llm=llm,
    verbose=True
)

# Define tasks with refined descriptions and expected outputs
research_task = Task(
    description="Conduct in-depth research on the latest Technology breakthroughs. Analyze their potential impact on various industries and society as a whole. Identify a topic that would captivate readers and provide valuable insights.",
    agent=researcher,
    expected_output="A comprehensive summary of cutting-edge tech developments, focusing on one breakthrough topic. Include its technical aspects, real-world applications, and potential future implications.",
    max_iterations=10,  # Increase iteration limit
    timeout=300  # Increase time limit
)

writing_task = Task(
    description="Transform the research findings into a compelling 1200-word blog post. Use analogies, real-world examples, and a conversational tone to make the content relatable and engaging.",
    agent=writer,
    expected_output="A captivating 1200-word blog post that hooks readers from the first sentence, explains the topic clearly, and leaves the audience inspired about the future of technology.",
    max_iterations=10,  # Increase iteration limit
    timeout=300  # Increase time limit
)

editing_task = Task(
    description="Refine the blog post for clarity, coherence, and impact. Ensure technical accuracy while maintaining accessibility for a general audience. Optimize the structure for online readability.",
    agent=editor,
    expected_output="A polished, error-free blog post with improved flow, precise language, and enhanced engagement factors such as subheadings, pull quotes, or callout boxes.",
    max_iterations=10,  # Increase iteration limit
    timeout=300  # Increase time limit
)

formatting_task = Task(
    description="Transform the edited content into an visually appealing HTML format. Use modern web design principles to enhance readability and engagement. Incorporate appropriate tags for SEO and accessibility.ensure all the contents are incoporated",
    agent=formatter,
    expected_output="A beautifully formatted HTML version of the blog post, featuring a clean layout, strategic use of white space, and CSS styles that enhance the reading experience across devices.do not",
    max_iterations=10,  # Increase iteration limit
    timeout=300  # Increase time limit
)

publishing_task = Task(
    description="Prepare the blog post for publication and promotion. Create an irresistible teaser that will drive clicks. Suggest meta descriptions, keywords, and social media snippets to boost visibility.",
    agent=publisher,
    expected_output="A publication-ready package including the formatted blog post, a compelling 60-word teaser, SEO elements, and social media promotional content to maximize the post's reach and engagement.",
    max_iterations=10,  # Increase iteration limit
    timeout=300  # Increase time limit
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

        # Check if the expected HTML tags are present
        if '<h1>' not in content or '</h1>' not in content:
            logging.error("Expected HTML tags not found in content.")
            return None

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

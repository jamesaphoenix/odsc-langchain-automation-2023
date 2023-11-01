import google.auth
from google.auth import impersonated_credentials
from google.auth.transport.requests import Request
import json
from langchain.llms import OpenAI
from langchain.prompts import (
    PromptTemplate,
)
from langchain.chains import LLMChain, SequentialChain
from langchain.prompts import PromptTemplate
import os
import requests
from pydantic import BaseModel, Field
from typing import Optional, List
import re
import uuid
import requests
import shutil
from PIL import Image

from custom_types import (
    Author,
    Brief,
    CharacterList,
    Dialogue,
    MarketingStrategy,
    Skill,
)
from strapi_crud import (
    create_skill,
    create_brief_component,
    create_dialogue_component,
    publish_simulator_as_draft,
    upload_image_to_strapi,
)
from utils import (
    extract_single_free_text_exercise,
    extract_single_multiple_choice_exercise,
    get_company_description,
    find_character,
    format_mapping,
    upload_blob,
    slugify,
)

import requests

# LangSmith:
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = f"HIDDEN_PROJECT"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

llm = OpenAI(temperature=0.7, max_tokens="500")  # type: ignore
better_llm = OpenAI(temperature=0.7, max_tokens="500", best_of=3)  # type: ignore

API_TOKEN: str = os.getenv("API_TOKEN")
BASE_URL = "https://cms.vexpower.com"
headers = {"Content-Type": "application/json", "Authorization": f"Bearer {API_TOKEN}"}

# The email address of the target service account you want to impersonate
target_service_account_email = (
    "HIDDEN_SERVICE_ACCOUNT_EMAIL"
)

# Double checking for the environment variables and using the .env file if unset:
if not os.getenv("LANGCHAIN_API_KEY"):
    raise ValueError(
        "Please set the LANGCHAIN_API_KEY environment variable to your LangSmith API key."
    )

if not os.getenv("API_TOKEN"):
    raise ValueError(
        "Please set the API_TOKEN environment variable to your Strapi API token."
    )

# Get the Application Default Credentials
source_credentials, project_id = google.auth.default()

# Create the impersonated credentials object for the target service account, bucket reading:
target_scopes = ["https://www.googleapis.com/auth/devstorage.read_write"]

impersonated_credentials = impersonated_credentials.Credentials(
    source_credentials=source_credentials,
    target_principal=target_service_account_email,
    target_scopes=target_scopes,
)

# Refresh the impersonated credentials
impersonated_credentials.refresh(Request())


def load_file(filename):
    with open(filename, "r") as f:
        return f.read()


def load_example_data():
    skill_name = "OpenAI Fine-Tuning"
    skill_description = "It's possible to fine-tune ChatGPT on your own data, so that it can write AI content that sounds like you."
    skill_link = "https://platform.openai.com/docs/guides/fine-tuning"
    title = "I want ChatGPT to write in my exact style."
    course_description = "The content you get from ChatGPT sounds dull and robotic. Editing AI content in your voice is tedious."
    access = "Free"
    role = "Developer"
    category = "Content"
    character = "Gustav Gieger"
    template_filename = "fine-tune.ipynb"
    template_url = "https://drive.google.com/file/d/13mk1liTcao3WZ4f65G92igIJRZqzWQlq/view?usp=drive_link"
    course_image = "scripts/temp_data/image.png"
    course_video = "scripts/temp_data/video.mp4"
    transcript = load_file("scripts/temp_data/transcript.txt")
    author = "Mike"

    data = (
        skill_name,
        skill_description,
        skill_link,
        title,
        course_description,
        access,
        role,
        category,
        character,
        template_filename,
        template_url,
        course_image,
        course_video,
        transcript,
        author,
    )
    return data

character_names = [
    "Gustav Gieger",
    "Dushyant Dixit",
    "Charlotte Cook",
    "Sam Smirnov",
    "William Winters",
    "Alexandra Anderson",
    "Danielle Oscar",
    "Ashton Donaghy",
    "Sally Valentine",
    "William Maple",
    "Margaret Nolan",
]
category_names = [
    "Advertising",
    "Attribution",
    "Content",
    "Retention",
    "Virality",
    "Conversion",
]

category_values = {
    "Advertising": MarketingStrategy.Advertising.value,
    "Attribution": MarketingStrategy.Attribution.value,
    "Content": MarketingStrategy.Content.value,
    "Retention": MarketingStrategy.Retention.value,
    "Virality": MarketingStrategy.Virality.value,
    "Conversion": MarketingStrategy.Conversion.value,
}


def get_characters():
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_TOKEN}",
    }
    url = "https://cms.vexpower.com/api/characters"
    response = requests.get(url, headers=headers)
    characters = CharacterList.parse_obj({"characters": response.json()["data"]})
    # character_names = [char.attributes.Name for char in characters.characters]
    return characters


def download_image_and_video_file(image, video):
    # if there isn't a course-data folder, create one
    if not os.path.exists("course-data"):
        os.makedirs("course-data")

    # move the video and image files to the course-data folder
    img = Image.fromarray(image.astype("uint8"))
    img.save("course-data/image.png")

    shutil.move(video, "course-data/video.mp4")


def run_chains(
    skill_name,
    skill_description,
    skill_link,
    title,
    course_description,
    access,
    role,
    category,
    character,
    template_filename,
    template_url,
    course_image,
    course_video,
    transcript,
    author,
):
    print("Downloading image and video...")
    download_image_and_video_file(course_image, course_video)

    print("Running all chains...")

    # Find a .mp4 file in the current working directory, it could be called anything with an .mp4 extension:
    video_file = [f for f in os.listdir("./course-data") if f.endswith(".mp4")][0]
    # video_file = course_video.name

    # Remove any special characters to make this into a safe file name:
    safe_video_file_name = slugify(skill_name)

    # Upload the video to the bucket:
    video_url = upload_blob(
        source_file_name=f"./course-data/{video_file}",
        destination_blob_name=f"videos/{safe_video_file_name}",
    )

    # Upload the image:
    image_extensions = [
        "jpeg",
        "jpg",
        "png",
        "gif",
        "bmp",
        "ico",
        "tiff",
        "tif",
        "webp",
    ]

    # Find an image file in the current working directory, it could be called anything with an image extension:
    image_file = [
        f
        for f in os.listdir("./course-data")
        if f.split(".")[-1].lower() in image_extensions
    ][0]
    # image_file = course_image.name

    # Remove any special characters to make this into a safe file name:
    safe_image_file_name = slugify(skill_name)

    # Upload the image to the bucket:
    image_url = upload_blob(
        source_file_name=f"./course-data/{image_file}",
        destination_blob_name=f"images/{safe_image_file_name}",
    )

    image_id = upload_image_to_strapi(
        f"./course-data/{image_file}",
        image_name=safe_image_file_name,
        API_TOKEN=API_TOKEN,
    )

    characters = get_characters()
    raw_character = find_character(character, characters)

    character = {
        "name": raw_character.attributes.Name,
        "job_title": raw_character.attributes.JobTitle,
        "company_name": raw_character.attributes.Company,
        "company_description": get_company_description(raw_character.id),
    }

    with open("scripts/prompts/exercises/free_text_exercise.txt", "r") as f:
        free_text_template = f.read()

    with open("scripts/prompts/exercises/multiple_choice_template.txt", "r") as f:
        multiple_choice_template = f.read()

    with open("scripts/prompts/exercises/key_learnings.txt", "r") as f:
        key_learnings_template = f.read()

    with open("scripts/prompts/exercises/prioritization.txt", "r") as f:
        prioritization_template = f.read()

    prompt_template = PromptTemplate(
        input_variables=["transcript"], template=free_text_template
    )
    free_text_exercises_chain = LLMChain(
        llm=llm, prompt=prompt_template, output_key="free_text_exercises"
    )

    prompt_template = PromptTemplate(
        input_variables=["transcript"], template=multiple_choice_template
    )
    multiple_choice_exercises_chain = LLMChain(
        llm=llm, prompt=prompt_template, output_key="multiple_choice_exercises"
    )

    prompt_template = PromptTemplate(
        input_variables=["transcript"], template=key_learnings_template
    )
    key_learnings_chain = LLMChain(
        llm=llm, prompt=prompt_template, output_key="key_learnings"
    )

    prompt_template = PromptTemplate(
        input_variables=[
            "free_text_exercises",
            "multiple_choice_exercises",
            "key_learnings",
            "title",
            "skill_name",
        ],
        template=prioritization_template,
    )
    prioritization_chain = LLMChain(
        llm=better_llm, prompt=prompt_template, output_key="prioritized_exercises"
    )

    overall_chain = SequentialChain(
        chains=[
            free_text_exercises_chain,
            multiple_choice_exercises_chain,
            key_learnings_chain,
            prioritization_chain,
        ],
        input_variables=["transcript", "title", "skill_name"],
        output_variables=[
            "free_text_exercises",
            "multiple_choice_exercises",
            "key_learnings",
            "prioritized_exercises",
        ],
        verbose=True,
    )

    result = overall_chain(
        {
            "transcript": transcript,
            "title": title,
            "skill_name": skill_name,
        }
    )

    prioritized_exercises = json.loads(result["prioritized_exercises"])
    final_exercises = []
    for exercise in prioritized_exercises:
        # If exercise_type is free_text, extract the exercise from free_text_exercises
        if exercise["exercise_type"] == "free_text":
            exercise_text = extract_single_free_text_exercise(
                result["free_text_exercises"], int(exercise["exercise_number"])
            )
        elif exercise["exercise_type"] == "multiple_choice":
            exercise_text = extract_single_multiple_choice_exercise(
                result["multiple_choice_exercises"], int(exercise["exercise_number"])
            )
        else:
            raise ValueError(f"Unknown exercise type: {exercise['exercise_type']}")

        final_exercises.append(
            {
                "exercise": exercise_text.strip(),
                "reason_for_choosing": exercise["reason_for_choosing"],
                "exercise_type": exercise["exercise_type"],
            }
        )

    # Transcript summary:
    with open("scripts/prompts/generic/summarize-video-transcript.txt", "r") as f:
        transcript_summary_template = f.read()

    # Slugline
    with open("scripts/prompts/tutorial/slugline.txt", "r") as f:
        slugline_template = f.read()

    # Action:
    with open("scripts/prompts/tutorial/action-description.txt", "r") as f:
        action_template = f.read()

    # # Dialogue:
    with open("scripts/prompts/tutorial/dialogue.txt", "r") as f:
        dialogue_template = f.read()

    # Brief:
    with open("scripts/prompts/tutorial/brief.txt", "r") as f:
        brief_template = f.read()

    prompt_template = PromptTemplate(
        input_variables=["transcript"],
        template=transcript_summary_template,
    )

    transcript_chain = LLMChain(
        llm=llm,
        prompt=prompt_template,
        output_key="video_transcript_summary",
    )

    prompt_template = PromptTemplate(
        input_variables=[
            "skill_name",
            "title",
            "character_name",
            "job_title",
            "company_name",
            "company_description",
            "video_transcript_summary",
        ],
        template=slugline_template,
    )

    slugline_chain = LLMChain(
        llm=llm,
        prompt=prompt_template,
        output_key="slugline",
    )

    prompt_template = PromptTemplate(
        input_variables=[
            "skill_name",
            "title",
            "character_name",
            "job_title",
            "company_name",
            "company_description",
            "video_transcript_summary",
            "slugline",
        ],
        template=action_template,
    )

    action_chain = LLMChain(llm=llm, prompt=prompt_template, output_key="action")

    prompt_template = PromptTemplate(
        input_variables=[
            "action",
            "skill_name",
            "title",
            "character_name",
            "job_title",
            "company_name",
            "company_description",
            "video_transcript_summary",
            "slugline",
        ],
        template=dialogue_template,
    )

    dialogue_chain = LLMChain(llm=llm, prompt=prompt_template, output_key="dialogue")

    tutorial_overall_chain = SequentialChain(
        chains=[transcript_chain, slugline_chain, action_chain, dialogue_chain],
        input_variables=[
            "skill_name",
            "title",
            "character_name",
            "job_title",
            "company_description",
            "company_name",
            "transcript",
        ],
        output_variables=["video_transcript_summary", "slugline", "action", "dialogue"],
        verbose=True,
        return_all=True,
    )

    tutorial_result = tutorial_overall_chain(
        {
            "skill_name": skill_name,
            "title": title,
            "transcript": transcript,
            "job_title": character["job_title"],
            "character_name": character["name"],
            "company_name": character["company_name"],
            "company_description": character["company_description"],
        }
    )

    # Brief:
    prompt_template = PromptTemplate(
        input_variables=["transcript", "skill_name"],
        template=brief_template,
    )

    brief_chain = LLMChain(llm=llm, prompt=prompt_template, output_key="brief")

    brief_result = brief_chain(
        {
            "skill_name": skill_name,
            "transcript": transcript,
        }
    )

    with open("course-data/course-outline.json", "w") as f:
        current_payload = {
            "brief": brief_result["brief"].strip(),
            "slugline": tutorial_result["slugline"].strip(),
            "action": tutorial_result["action"]
            .strip()
            .replace(" Let's get started!", ""),
            "dialogue": tutorial_result["dialogue"].strip(),
            "final_exercises": final_exercises,
        }
        json.dump(current_payload, f, indent=4)

    with open("course-data/course-outline.json", "r") as f:
        course_outline = json.load(f)

    # Creating the components:
    brief = Brief(
        title=skill_name,
        text=course_outline["brief"],
        Image=image_id,
    )

    # Dialogue
    dialogue = Dialogue(
        DialogueText=course_outline["dialogue"],
        Slugline=course_outline["slugline"],
        Action=course_outline["action"],
        Character=raw_character.id,
    )

    # Skill
    skill = Skill(
        Name=skill_name, Description=skill_description, Link=skill_link, Image=image_id
    )

    brief_component = create_brief_component(brief)
    dialogue_component = create_dialogue_component(dialogue)
    skill_id = create_skill(skill.dict(), headers)

    category_id = category_values[category]

    mapping = {
        "courseID": str(uuid.uuid4()),
        "DialogueSectionID": str(uuid.uuid4()),
        "VideoSectionID": str(uuid.uuid4()),
        "FileSectionID": str(uuid.uuid4()),
        "BriefSectionID": str(uuid.uuid4()),
        "chapterID": str(uuid.uuid4()),
        "Design": "Dossier",
        "Title": title,
        "Description": course_description,
        "Role": role,
        "Access": access,
        "HasReviewed": False,
        "Image": image_id,
        "FileName": template_filename,
        "FileURL": template_url,
        "VideoFileURL": video_url,
        "VideoTranscription": transcript,
        "Category": category_id,
        "Skill": skill_id,
        "Dialogue": dialogue_component,
        "Brief": brief_component,
        # "Author": - Added later
        # "LogoImages": - Added later
        # "Exercises # - Added later
    }

    brief_text = course_outline["brief"]
    slugline_text = course_outline["slugline"]
    action_text = course_outline["action"]
    dialogue_text = course_outline["dialogue"]

    if author == "James":
        author_id = Author.James
    else:
        author_id = Author.Mike

    final_mapping = format_mapping(mapping, author_id)

    with open("course-data/final-mapping.json", "w") as f:
        json.dump(final_mapping, f, indent=4)

    return (
        image_url,
        video_url,
        brief_text,
        slugline_text,
        action_text,
        dialogue_text,
        final_exercises,
    )


def publish_course():
    with open("course-data/course-outline.json", "r") as f:
        course_outline = json.load(f)

    with open("course-data/final-mapping.json", "r") as f:
        final_mapping = json.load(f)

    response, published_course = publish_simulator_as_draft(
        final_mapping, headers=headers
    )
    publish_exercises(course_outline, published_course)
    return response, published_course


class Exercise(BaseModel):
    ExerciseName: str = Field(..., max_length=255)
    ExerciseSectionID: str
    Format: str
    SimulatorCourses: List[int]
    Hint: Optional[str]
    Answer: Optional[str]
    ImageURL: Optional[str]
    ImageSource: Optional[str]
    ExerciseOptions: Optional[List[dict]] = []


def parse_multiple_choice_exercise(exercise_dict, course_id) -> Exercise:
    # Extract the question number and content:
    exercise_number, question = re.match(
        r"(\d+)\) Question: (.*?)\n[A-Z]\)", exercise_dict["exercise"], re.DOTALL
    ).groups()

    # Extract the answers from the text:
    answer_options = re.findall(
        r"\n([A-Z])\) (.*?)(?=\n[A-Z]\)|$)", exercise_dict["exercise"], re.DOTALL
    )

    # Find the correct answer:
    correct_answer = re.search(r"Answer: ([A-Z])", exercise_dict["exercise"]).group(1)

    options = []
    for letter, content in answer_options:
        pruned_content = re.sub(r"\nAnswer: [A-Z]\).*$", "", content, flags=re.DOTALL)
        option = {
            "__component": "simulator-data.exercise-options-component",
            "Option": pruned_content,
            "IsCorrectAnswer": (letter == correct_answer),
        }
        options.append(option)

    return Exercise(
        ExerciseName=question,
        ExerciseSectionID=str(uuid.uuid4()),
        Format="choice",
        SimulatorCourses=[course_id],
        Answer=correct_answer,
        ExerciseOptions=options,
        Hint="",
        ImageURL="",
        ImageSource="",
    )


def parse_free_text_exercise(exercise_dict, course_id) -> Exercise:
    # Extract the question number, question, and answer from the text:
    exercise_number, question, answer = re.match(
        r"(\d+)\) Question: (.*?)\nAnswer: (.*)", exercise_dict["exercise"], re.DOTALL
    ).groups()

    free_text = {
        "__component": "simulator-data.exercise-options-component",
        "Option": answer,
        "IsCorrectAnswer": True,
    }
    return Exercise(
        ExerciseName=question,
        ExerciseSectionID=str(uuid.uuid4()),
        Format="freeText",
        SimulatorCourses=[course_id],
        Answer=answer,
        ExerciseOptions=[free_text],
        Hint="",
        ImageURL="",
        ImageSource="",
    )


def publish_exercises(course_outline, published_course):
    payloads = []

    for exercise in course_outline["final_exercises"]:
        if exercise["exercise_type"] == "multiple_choice":
            payloads.append(parse_multiple_choice_exercise(exercise, published_course))
        else:
            payloads.append(parse_free_text_exercise(exercise, published_course))

    for index, payload in enumerate(payloads):
        response = requests.post(
            f"{BASE_URL}/api/exercises", json={"data": payload.dict()}, headers=headers
        ).json()
        print(index, response)

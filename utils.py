from custom_types import Author, AuthorLogoImages, CharacterList
from google.cloud import storage
import json
import re
import uuid


def format_mapping(mapping: dict, person: Author) -> dict:
    print(person)
    if person == Author.James:
        mapping["LogoImages"] = AuthorLogoImages.James.value["strapi_ids"]
        mapping["Author"] = Author.James.value
    elif person == Author.Mike:
        mapping["LogoImages"] = AuthorLogoImages.Mike.value["strapi_ids"]
        mapping["Author"] = Author.Mike.value
    else:
        raise ValueError("Unknown author")
    return mapping


def find_character(character_name, character_list: CharacterList):
    for character in character_list.characters:
        if character.attributes.Name == character_name:
            return character
    raise ValueError(f"Character {character_name} not found")


def extract_single_free_text_exercise(free_text_exercises: str, question_number: int):
    free_text_exercises_match = re.findall(
        r"(\d+\) Question: .+?\nAnswer: .+?)(?=\d+\) Question: |\Z)",
        free_text_exercises,
        re.DOTALL,
    )
    if not free_text_exercises_match:
        raise ValueError("No free text exercises found")
    # Find the question number:
    for exercise in free_text_exercises_match:
        if exercise.startswith(f"{question_number}"):
            return exercise
    raise ValueError("No free text exercise found with that number")


def extract_single_multiple_choice_exercise(
    multiple_choice_exercises: str, question_number: int
) -> str:
    multiple_choice_exercises_match = re.findall(
        r"(\d+\) Question: .+?\nAnswer: .+?)(?=\d+\) Question: |\Z)",
        multiple_choice_exercises,
        re.DOTALL,
    )
    if not multiple_choice_exercises_match:
        raise ValueError("No multiple choice exercises found")
    # Find the question number:
    for exercise in multiple_choice_exercises_match:
        if exercise.startswith(f"{question_number}"):
            return exercise
    raise ValueError("No multiple choice exercise found with that number")


def upload_blob(
    source_file_name, destination_blob_name, bucket_name="strapi_cms_assets", project_id="vexpower-2b2c5"
):
    """Uploads a file to the bucket."""
    storage_client = storage.Client(project=project_id)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)

    full_path = f"https://storage.googleapis.com/{bucket_name}/{destination_blob_name}"
    print("File {} uploaded to {}.".format(source_file_name, full_path))

    return full_path


def slugify(name: str) -> str:
    return (
        "".join([c for c in name if c.isalpha() or c.isdigit() or c == " "])
        .rstrip()
        .replace(" ", "-")
        .lower()
        + "-"
        + str(uuid.uuid4())
        + ".mp4"
    )


def get_company_description(id: int) -> str:
    """
    Function to return the Company Description given an ID.

    Args:
    id (int): The ID of the individual whose Company Description is required.
    data (str): The JSON string containing the data.

    Returns:
    str: The Company Description if found else an appropriate message.
    """
    with open("scripts/temp_data/character.json", "r") as f:
        characters = json.load(f)

    # Search for the individual with the given ID
    for character in characters:
        if character["id"] == id:
            return character["attributes"]["CompanyDescription"]

    raise ValueError(f"ID {id} not found")

from scripts.custom_types import Brief, Dialogue, Skill
import requests


def upload_image_to_strapi(image_filepath: str, image_name, API_TOKEN: str) -> str:
    with open(image_filepath, "rb") as f:
        image_content = f.read()

    # Uploading an image file:
    files = {
        "files": (
            image_name,
            image_content,
            "image",
            {
                "Authorization": f"Bearer {API_TOKEN}",
            },
        )
    }
    response = requests.post(
        "https://cms.vexpower.com/api/upload",
        files=files,
        headers={"Authorization": f"Bearer {API_TOKEN}"},
    )
    print(response.json())
    id_ = response.json()[0]["id"]
    return id_


def create_skill(skill: Skill, headers):
    resp = requests.post(
        "https://cms.vexpower.com/api/skills", json={"data": skill}, headers=headers
    )
    print(resp.json())
    return resp.json()["data"]["id"]


def create_brief_component(brief: Brief) -> dict:
    """
    Creates the brief component.
    """
    brief_component = {
        "__component": "simulator-data.brief-component",
        "Title": brief.title,
        "Text": brief.text,
    }

    if brief.Image:
        brief_component["Image"] = brief.Image  # type: ignore
    if brief.imageSource:
        brief_component["ImageSource"] = brief.imageSource
    return brief_component


def create_dialogue_component(dialogue: Dialogue) -> dict:
    """
    Creates the dialogue component.
    """
    dialogue_component = {
        "__component": "simulator-data.dialogue-component",
        "DialogueText": dialogue.DialogueText,
        "Slugline": dialogue.Slugline,
        "Character": dialogue.Character,
        "Action": dialogue.Action,
    }
    return dialogue_component


def publish_simulator_as_draft(payload: dict, headers: dict) -> int:
    """
    Publishes the simulator as a draft to the Strapi database.
    """
    draft_payload = {**payload, "publishedAt": None}

    response = requests.post(
        f"https://cms.vexpower.com/api/sims?populate=*",
        json={"data": draft_payload},
        headers=headers,
    ).json()

    print(response)
    try:
        simulator_id = response["data"]["id"]
    except KeyError:
        simulator_id = None

    return response, simulator_id

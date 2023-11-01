from enum import Enum
from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field
from typing import Optional, List


class MarketingStrategy(Enum):
    Attribution = 1
    Advertising = 2
    Virality = 3
    Content = 4
    Sales = 5
    Conversion = 6
    Retention = 7


class Access(Enum):
    Free = "Free"
    Premium = "Premium"


class Author(Enum):
    James = 1
    Mike = 2


class AuthorLogoImages(Enum):
    James = {
        "strapi_ids": [1515, 1516, 1517],
    }

    Mike = {
        "strapi_ids": [1512, 1513, 1514],
    }


class Skill(BaseModel):
    Name: str = Field(..., max_length=255)
    Description: str = Field(..., max_length=1000)
    Link: str
    Image: Optional[int] = None # noqa


class Format(str, Enum):
    freeText = "freeText"
    choice = "choice"


class Brief(BaseModel):
    title: str = Field(..., max_length=255)
    text: str = Field(...)
    Image: Optional[int] = None # noqa
    imageURL: Optional[str] = None # noqa
    imageSource: Optional[str] = None # noqa


class Dialogue(BaseModel):
    DialogueText: Optional[str] = Field(...)
    Slugline: str = Field(..., max_length=1000)
    Character: int = Field(...)
    Action: str = Field(..., max_length=1000)


class Attributes(BaseModel):
    characterID: str
    Name: str
    JobTitle: str
    Company: str
    Email: str
    createdAt: datetime
    updatedAt: datetime
    publishedAt: datetime
    NickName: str
    CompanyDescription: Optional[str] = None # noqa


class Character(BaseModel):
    id: int
    attributes: Attributes # noqa


class CharacterList(BaseModel):
    characters: List[Character]


class Exercise(BaseModel):
    ExerciseName: str = Field(..., max_length=255)
    ExerciseSectionID: str
    Format: str
    SimulatorCourses: List[int]
    Hint: Optional[str] = None # noqa
    Answer: Optional[str] = None # noqa
    ImageURL: Optional[str] = None # noqa
    ImageSource: Optional[str] = None # noqa
    ExerciseOptions: Optional[List[dict]] = []

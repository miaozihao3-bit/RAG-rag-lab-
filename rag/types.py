from dataclasses import dataclass
from typing import List


@dataclass
class Chunk:
    id: str
    source: str
    heading: str
    text: str
    start: int
    end: int


@dataclass
class Hit:
    chunk: Chunk
    score: float


@dataclass
class Source:
    source: str
    heading: str
    score: float
    text: str


@dataclass
class Answer:
    text: str
    sources: List[Source]
    refused: bool
    latency_ms: int

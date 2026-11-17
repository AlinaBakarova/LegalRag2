from pydantic import BaseModel, ConfigDict, Field


class CitationDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source_id: str = Field(pattern=r"^S\d+$")
    quote: str = Field(min_length=10, max_length=350)


class ClaimDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=1, max_length=600)
    citations: list[CitationDraft] = Field(min_length=1, max_length=2)


class AnswerDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sufficient: bool
    claims: list[ClaimDraft] = Field(max_length=3)
    limitations: list[str] = Field(max_length=4)


class SupportReview(BaseModel):
    model_config = ConfigDict(extra="forbid")
    supported: bool
    unsupported_claims: list[int] = Field(max_length=3)

from sqlalchemy import Text
from sqlalchemy.orm import Mapped, mapped_column

from .database import Base


class ExerciseChunk(Base):

    __tablename__ = "exercise_chunks"

    id: Mapped[str] = mapped_column(primary_key=True)

    page_content: Mapped[str] = mapped_column(Text)

    week: Mapped[str | None]

    day: Mapped[str | None]
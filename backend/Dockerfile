FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends tesseract-ocr tesseract-ocr-ind \
    && rm -rf /var/lib/apt/lists/*

ENV UV_PROJECT_ENVIRONMENT=/opt/venv UV_NO_CACHE=1 UV_LINK_MODE=copy \
    PATH=/opt/venv/bin:$PATH PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /app

# uv is only needed to install; it is removed in the same layer to keep the image small.
COPY pyproject.toml uv.lock ./
RUN pip install --no-cache-dir uv \
    && uv sync --frozen --no-dev --no-install-project \
    && pip uninstall -y uv

COPY app app
COPY migrations migrations

# Run as an unprivileged user: the worker parses hostile files, the api answers users. The code and the venv stay
# root-owned and read-only; the only writable place is the uploads mount (see compose.yaml).
RUN useradd --system --uid 10001 --user-group --no-create-home --home-dir /nonexistent --shell /usr/sbin/nologin app
USER 10001:10001

EXPOSE 8000
CMD ["uvicorn", "app.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]

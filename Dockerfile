# The review screen (M5) with the committed results. The survey data is optional: mount it to see camera frames.
#   docker build -t logbook-subsea .
#   docker run --rm -p 127.0.0.1:8000:8000 -v "$PWD/data:/srv/data:ro" -v "$PWD/runs:/srv/runs" logbook-subsea
# Then open http://127.0.0.1:8000. The app has no login, so keep the port on 127.0.0.1.
FROM python:3.12-slim
WORKDIR /srv
COPY requirements-app.txt .
RUN pip install --no-cache-dir -r requirements-app.txt
COPY app app
COPY docs/results docs/results
RUN useradd --create-home reviewer && mkdir runs && chown reviewer runs
USER reviewer
EXPOSE 8000
CMD ["uvicorn", "app.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]

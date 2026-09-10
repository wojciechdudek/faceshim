# syntax=docker/dockerfile:1
# ORT_PACKAGE=onnxruntime==1.29.0  -> CPU (default)
# ORT_PACKAGE=onnxruntime-openvino -> Intel iGPU/CPU via OpenVINO (also pass /dev/dri and ORT_PROVIDERS at runtime)
ARG ORT_PACKAGE=onnxruntime==1.29.0
ARG MODEL_NAME=buffalo_l

FROM python:3.11-slim AS build
ARG ORT_PACKAGE
ARG MODEL_NAME
RUN apt-get update && apt-get install -y --no-install-recommends build-essential g++ \
    && rm -rf /var/lib/apt/lists/*
ENV VIRTUAL_ENV=/opt/venv PATH=/opt/venv/bin:$PATH
RUN python -m venv /opt/venv && pip install --no-cache-dir --upgrade pip wheel Cython
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt "${ORT_PACKAGE}"
# Bake the model pack into the image so the container never downloads at runtime.
RUN python -c "from insightface.app import FaceAnalysis; \
    FaceAnalysis(name='${MODEL_NAME}', providers=['CPUExecutionProvider'], allowed_modules=['detection','recognition']).prepare(ctx_id=-1)"

FROM python:3.11-slim
ARG MODEL_NAME
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 libgomp1 gosu \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 app
COPY --from=build /opt/venv /opt/venv
COPY --from=build --chown=app:app /root/.insightface /home/app/.insightface
WORKDIR /app
COPY --chown=app:app app.py selfcheck.py entrypoint.sh ./
RUN mkdir -p /data && chown app:app /data
ENV PATH=/opt/venv/bin:$PATH HOME=/home/app MODEL_NAME=${MODEL_NAME} DATA_DIR=/data \
    PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2
# No USER here: entrypoint.sh fixes ownership of a bind-mounted /data, then drops to `app`.
VOLUME ["/data"]
EXPOSE 5000
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if b'true' in urllib.request.urlopen('http://127.0.0.1:5000/', timeout=4).read() else 1)"
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5000", "--no-access-log"]

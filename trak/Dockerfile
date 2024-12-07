FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04
SHELL ["/bin/bash", "-c"]
COPY apt_minimal.sh .
RUN ./apt_minimal.sh && rm apt_minimal.sh
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    source $HOME/.local/bin/env && \
    uv python install 3.12 && \
    uv venv -p 3.12 && \
    source .venv/bin/activate && \
    uv pip install torch && \
    uv pip install traker[fast] --no-build-isolation
COPY requirements.txt .
COPY training /git/pdpl/training/
RUN source $HOME/.local/bin/env && \
    uv pip install -r requirements.txt
RUN echo '#!/bin/bash' > /entrypoint.sh && \
    echo 'if [ $# -eq 0 ]; then' >> /entrypoint.sh && \
    echo '    sleep infinity' >> /entrypoint.sh && \
    echo 'else' >> /entrypoint.sh && \
    echo '    exec "$@"' >> /entrypoint.sh && \
    echo 'fi' >> /entrypoint.sh && \
    chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
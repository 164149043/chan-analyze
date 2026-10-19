FROM python:3.11-slim

ENV TZ=Asia/Shanghai \
    LANG=C.UTF-8 \
    PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        wget \
        curl \
        tzdata \
    && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir

COPY src /app/src
COPY web /app/web

WORKDIR /app/web/chanlun_chart

EXPOSE 9900

# 数据目录挂载（K线缓存/缠论数据）：-v /path/to/.chanlun_pro:/root/.chanlun_pro
VOLUME ["/root/.chanlun_pro"]

CMD ["gunicorn", "-w", "1", "-k", "gthread", "--threads", "4", "-b", "0.0.0.0:9900", "app:app"]

# pytdx 1.72r1 修复版（PyPI 1.72 有 TdxFunctionCallError bug），覆盖安装
COPY vendor/pytdx /usr/local/lib/python3.11/site-packages/pytdx
COPY vendor/pytdx-1.72r1.dist-info /usr/local/lib/python3.11/site-packages/pytdx-1.72r1.dist-info

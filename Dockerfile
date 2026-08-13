FROM node:20-bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    pkg-config \
    python3 \
    libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY package.json ./

# Use distro OpenCV instead of compiling OpenCV from source during npm install.
ENV OPENCV4NODEJS_DISABLE_AUTOBUILD=1

RUN npm install --omit=dev

COPY server.js new_template.jpg ./

ENV NODE_ENV=production
ENV PORT=3000
EXPOSE 3000

CMD ["npm", "start"]

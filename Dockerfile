FROM alpine:3.21
RUN apk add --no-cache ca-certificates tzdata
WORKDIR /app
COPY hindsight-relay-linux .
ENTRYPOINT ["/app/hindsight-relay-linux"]

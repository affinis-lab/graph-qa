FROM neo4j:3.5

ENV NEO4J_PIDFILE=/var/run/neo4j/neo4j.pid

COPY scripts/run_neo4j.sh .
RUN chmod +x run_neo4j.sh

RUN mkdir /var/run/neo4j
RUN touch /var/run/neo4j/neo4j.pid
RUN chmod 666 /var/run/neo4j/neo4j.pid

CMD [ "sh", "./run_neo4j.sh"]

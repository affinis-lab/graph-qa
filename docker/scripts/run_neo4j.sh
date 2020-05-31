neo4j start &
echo $! >> /var/run/neo4j/neo4j.pid
while true; do sleep 1000; done

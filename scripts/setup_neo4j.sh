alias dc="docker-compose -f ../docker/docker-compose.dev.yml"

dc up -d neo4j

dc exec neo4j neo4j stop

dc exec neo4j neo4j-admin import \
    --nodes=/data_in/paragraph.csv \
    --nodes=/data_in/sentence.csv \
    --relationships=/data_in/paragraph_preceded_by.csv \
    --relationships=/data_in/paragraph_followed_by.csv \
    --relationships=/data_in/paragraph_summarized_by.csv \
    --relationships=/data_in/paragraph_links_to.csv \
    --relationships=/data_in/sentence_of.csv \
    --relationships=/data_in/sentence_preceded_by.csv \
    --relationships=/data_in/sentence_followed_by.csv \
    --relationships=/data_in/sentence_links_to.csv \
    --ignore-missing-nodes=true

dc exec neo4j chown -R neo4j:neo4j /data/databases/graph.db

# TODO: read neo4j admin password from env
#dc restart neo4j
# dc exec neo4j cypher-shell -u neo4j -p "password" "CREATE INDEX ON :Paragraph(paragraphId)"

dc stop neo4j

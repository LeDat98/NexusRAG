# Support Postgres as Vector DB

## Goal
Add an option to use PostgreSQL (via pgvector) as a vector database instead of just ChromaDB.

## Success Criteria
1. Configuration setting `VECTOR_DB_PROVIDER` (`chroma` or `postgres`) is added and properly respected.
2. Abstract VectorStore interface or a consistent duck-typing signature is defined.
3. Existing `VectorStore` in `vector_store.py` is renamed/moved to `ChromaVectorStore`.
4. A new `PostgresVectorStore` is created.
5. `pgvector` dependency is added to `backend/requirements.txt` and DB initialized properly with `pgvector` extension.
6. The factory methods provide the correct vector store instance.
7. Postgres vector database configuration is available in `docker-compose.yml`.

## Steps
- [ ] List directory structure and examine database models/migrations to understand how to add `pgvector`.
- [ ] Create detailed Implementation Plan (pseudocode) for approval.
- [ ] Update `backend/app/core/config.py`.
- [ ] Define abstract vector store and implement `ChromaVectorStore` & `PostgresVectorStore`.
- [ ] Update database setup (migrations or init scripts) to initialize `vector` extension and table if using PG.
- [ ] Update `backend/requirements.txt`.
- [ ] Update `docker-compose.yml` and related environment variables.
- [ ] Verify functionality (testing with both providers).

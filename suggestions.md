# Insightful Rerank Query Suggestions for Madison Speeches

This document provides recommended queries for demonstrating the semantic capabilities of both the `LocalReranker` and `RemoteReranker` using James Madison's presidential speeches (1809–1817).

James Madison's texts are written in formal, early 19th-century political prose. Simple keyword-based searches (like BM25) often fail when queried with modern terms, whereas semantic cross-encoders like **jina-reranker-v3** excel by mapping the conceptual meanings.

---

## 1. Separation of Church and State

*   **Query**: `"Separation of religious institutions from civil government authority"`
*   **Target Speech**: `06_1811-02-21_Veto_Alexandria_Protestant_Episcopal_Church.txt`
*   **Linguistic Gap**: 
    - Madison's speech never uses the phrase *"separation of church and state."* 
    - Instead, it uses formal phrasing: *"the essential distinction between Civil and Religious functions"* and *"Congress shall make no law respecting a Religious establishment."*
    - Keyword search fails because it would match other speeches repeating the words "civil" or "authority" in relation to military/legal matters.
    - The cross-encoder correctly identifies the conceptual separation of religious and civic jurisdiction, ranking this veto first.

---

## 2. Impressment of Sailors

*   **Query**: `"British cruisers kidnapping or forcing American citizens into military service"`
*   **Target Speech**: `08_1812-06-01_War_Message_Foreign_Policy_Crisis.txt`
*   **Linguistic Gap**: 
    - The speech uses the formal term **"impressment"** and describes the British cruisers' practice of *"violating the American flag"* to *"drag them on board their ships of war."*
    - The query uses the modern word **"kidnapping"**, which appears nowhere in the speeches. 
    - Semantic reranking understands that "kidnapping and forcing into military service" corresponds directly to "impressment," successfully surface-ranking the War Message.

---

## 3. Infrastructure & Constitutional Limits

*   **Query**: `"Lack of constitutional authority for federal funding of national infrastructure"`
*   **Target Speech**: `22_1817-03-03_Veto_Message_Internal_Improvements.txt`
*   **Linguistic Gap**: 
    - The word **"infrastructure"** did not exist in 1817. Madison refers instead to **"roads and canals"** and **"internal improvements."**
    - A traditional keyword search for "infrastructure" yields zero results.
    - Reranking bridges the 200-year linguistic gap, connecting the modern term "national infrastructure" to the vetoed internal improvements bill.

---

## 4. Monetary Policy & Inflation

*   **Query**: `"Financial stability and paper currency during military conflicts"`
*   **Target Speech**: `19_1815-01-30_Veto_Message_National_Bank.txt`
*   **Linguistic Gap**: 
    - Madison's text describes **"depreciated paper,"** **"public credit,"** **"treasury notes,"** and the **"subscriber bank."**
    - The query features highly modern economic concepts like "financial stability" and "monetary policy."
    - Semantic search associates "paper currency during military conflicts" with Madison's war-time veto explaining how a proposed national bank might impact currency depreciation.

---

## 5. Peace Treaties & Reconciliation

*   **Query**: `"Termination of hostilities and restoration of friendly international commerce"`
*   **Target Speech**: `18_1815-02-18_Special_Message_Treaty_of_Ghent.txt`
*   **Linguistic Gap**:
    - The speech celebrates the **Treaty of Ghent** and uses historical phrasing like **"pacification"** and **"interposition of Divine Providence."**
    - The reranker easily matches the query's conceptual focus on "restoration of friendly commerce" to Madison’s reflections on entering an era of peaceful trade and public prosperity.

---

## How to Test

You can test these queries directly by modifying the `QUERY` variable in `quick_test4.py`:

```python
# Query to search for
QUERY = "Separation of religious institutions from civil government authority"
```

Running the script with these semantic queries will demonstrate how the reranker effectively prioritizes highly relevant documents over ones that merely share surface-level vocabulary.

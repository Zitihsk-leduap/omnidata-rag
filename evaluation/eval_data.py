# PRODUCTION TEST DATA: Company Act 2063 Questions with expected answers
EVAL_QUERIES = [
    {
        "query": "How does the Investor Protection Fund work?",
        "query_ne": "लगानीकर्ता संरक्षण कोष कसरी काम गर्छ?",
        "expected_keywords": ["दफा १८३", "शेयर", "लगानी", "पाँच वर्ष"],
        "mode": "explanation",
    },
    {
        "query": "What is the minimum number of founders for a Public Company?",
        "query_ne": "पब्लिक कम्पनीको लागि न्यूनतम कति जना संस्थापक चाहिन्छ?",
        "expected_keywords": ["सातजना", "संस्थापक", "पब्लिक"],
        "expected_answer": "at least 7 founders",
        "mode": "fact",
    },
    {
        "query": "What happens to application money if share allotment fails?",
        "query_ne": "शेयर बाँडफाँड हुन नसकेमा दरखास्त रकमको के हुन्छ?",
        "expected_keywords": ["दरखास्त", "रकम", "फिर्ता"],
        "mode": "explanation",
    },
    {
        "query": "Can a company pay dividends before writing off accumulated losses?",
        "query_ne": "एउटा कम्पनीले सञ्चित नोक्सानी लेखेर हटाउनु अगाडी लाभांश दिन सक्छ?",
        "expected_keywords": ["लाभांश", "सञ्चित नोक्सानी", "पहिले"],
        "mode": "explanation",
    },
    {
        "query": "When did the Company Act 2063 come into force?",
        "query_ne": "कम्पनी ऐन २०६३ कहिले लागू भयो?",
        "expected_keywords": ["२०६३", "असोज", "प्रारम्भ"],
        "mode": "fact",
    },
]


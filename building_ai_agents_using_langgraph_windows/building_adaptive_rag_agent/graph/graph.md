```mermaid
---
config:
  theme: base
  themeVariables:
    fontSize: '18px'
    primaryColor: '#0e0d0dff'
    primaryTextColor: '#0bbd3eff'
    primaryBorderColor: '#0a0a0aff'
    lineColor: '#ffffff'
    tertiaryColor: '#050505ff'
  flowchart:
    curve: linear
    defaultRenderer: "elk"
---
graph TD;
	__start__([<p>__start__</p>]):::first
	question_rephraser_node(question_rephraser_node)
	retriever_node(retriever_node)
	llm_knowledge_node(llm_knowledge_node)
	web_search_node(web_search_node)
	pass_as_is_node(pass_as_is_node)
	document_grader_node(document_grader_node)
	generate_node(generate_node)
	hallucination_checker_node(hallucination_checker_node)
	generated_answer_checker_node(generated_answer_checker_node)
	__end__([<p>__end__</p>]):::last
	__start__ --> question_rephraser_node;
	document_grader_node -. &nbsp;relevant&nbsp; .-> pass_as_is_node;
	document_grader_node -. &nbsp;irrelevant&nbsp; .-> question_rephraser_node;
	generate_node --> generated_answer_checker_node;
	generate_node --> hallucination_checker_node;
	generated_answer_checker_node --> pass_as_is_node;
	hallucination_checker_node --> pass_as_is_node;
	llm_knowledge_node --> document_grader_node;
	pass_as_is_node -. &nbsp;no_of_resources_exhausted&nbsp; .-> __end__;
	pass_as_is_node -. &nbsp;generate&nbsp; .-> generate_node;
	pass_as_is_node -. &nbsp;llm_knowledge&nbsp; .-> llm_knowledge_node;
	pass_as_is_node -. &nbsp;failed&nbsp; .-> web_search_node;
	question_rephraser_node -. &nbsp;llm_knowledge&nbsp; .-> llm_knowledge_node;
	question_rephraser_node -. &nbsp;vector_store&nbsp; .-> retriever_node;
	question_rephraser_node -. &nbsp;web_search&nbsp; .-> web_search_node;
	retriever_node --> document_grader_node;
	web_search_node --> document_grader_node;
	pass_as_is_node -. &nbsp;route_llm_knowledge_or_web_search&nbsp; .-> pass_as_is_node;
	classDef default fill:#f2f0ff,line-height:1.2
	classDef first fill-opacity:0
	classDef last fill:#bfb6fc
```

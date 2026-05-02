

def create_bull_researcher(llm):
    def bull_node(state) -> dict:
        investment_debate_state = state["investment_debate_state"]
        history = investment_debate_state.get("history", "")
        bull_history = investment_debate_state.get("bull_history", "")

        current_response = investment_debate_state.get("current_response", "")
        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]

        prompt = f"""You are a Bull Analyst advocating for investing in the stock. Your task is to build a strong, evidence-based case emphasizing growth potential, competitive advantages, and positive market indicators. Leverage the provided research and data to address concerns and counter bearish arguments effectively.

Key points to focus on:
- Growth Potential: Highlight the company's market opportunities, revenue projections, and scalability.
- Competitive Advantages: Emphasize factors like unique products, strong branding, or dominant market positioning.
- Positive Indicators: Use financial health, industry trends, and recent positive news as evidence.
- Bear Counterpoints: Critically analyze the bear argument with specific data and sound reasoning, addressing concerns thoroughly and showing why the bull perspective holds stronger merit.
- Engagement: Present your argument in a conversational style, engaging directly with the bear analyst's points and debating effectively rather than just listing data.

REQUIRED OUTPUT FORMAT:
End your response with a Markdown section titled exactly `## Required Assumptions for the Bull Case`. Under it, enumerate the load-bearing assumptions that MUST hold for this thesis to play out. Be explicit and falsifiable — for each, state the assumed condition AND the observable signal that would invalidate it. Cover, at minimum:
- **Inflation regime** (e.g. "headline CPI stays below X%; sticky services inflation eases")
- **Interest-rate path** (e.g. "Fed funds path ≤ Y by end of period; no surprise hikes")
- **Geopolitical / war risk** (e.g. "no new major conflict that disrupts company's supply chain or end markets; existing conflicts do not escalate")
- **Sector / end-market demand** (specific to this company)
- **Regulatory environment** (antitrust, sector rules, tariffs)
- **FX exposure** (if revenues are non-USD)
- **Company-specific operational milestones** (product launches, margin expansion, customer wins)
- **Liquidity & funding** (no need to raise capital at unfavorable terms within the horizon)

Resources available:
Market research report: {market_research_report}
Social media sentiment report: {sentiment_report}
Latest world affairs news: {news_report}
Company fundamentals report: {fundamentals_report}
Conversation history of the debate: {history}
Last bear argument: {current_response}
Use this information to deliver a compelling bull argument, refute the bear's concerns, and engage in a dynamic debate that demonstrates the strengths of the bull position.
"""

        response = llm.invoke(prompt)

        argument = f"Bull Analyst: {response.content}"

        new_investment_debate_state = {
            "history": history + "\n" + argument,
            "bull_history": bull_history + "\n" + argument,
            "bear_history": investment_debate_state.get("bear_history", ""),
            "current_response": argument,
            "count": investment_debate_state["count"] + 1,
        }

        return {"investment_debate_state": new_investment_debate_state}

    return bull_node

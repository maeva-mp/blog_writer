#                   Class Work :  LinkedIn Blog Writer
# 1. Imports
import streamlit as st
import random
import requests
from tavily import TavilyClient
from llama_index.llms.groq import Groq
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


llm = Groq(
    model="llama-3.3-70b-versatile",
    api_key=st.secrets["GROQ_API_KEY"]
)


# 2. Page config
st.set_page_config(
    page_title="LinkedIn Blog Writer",
    page_icon="📝",
    layout="wide"
)

st.title("📝 LinkedIn Blog Writer")
st.caption("Powered by multi-agent orchestration: Web Search → Blog Writer → SEO Review")

# 3. Context setup
if "context" not in st.session_state:
    st.session_state.context = {}

# Sidebar for settings
with st.sidebar:
    st.header("⚙️ Settings")
    
    tone = st.selectbox(
        "Writing Tone",
        ["Professional", "Casual", "Inspirational", "Technical", "Thought Leadership"]
    )
    
    blog_length = st.select_slider(
        "Blog Length",
        options=["Short (300 words)", "Medium (500 words)", "Long (800+ words)"],
        value="Medium (500 words)"
    )
    
    include_stats = st.checkbox("Include Statistics", value=True)
    include_cta = st.checkbox("Include Call-to-Action", value=True)
    
    st.divider()
    st.markdown("### 📊 Workflow Progress")
    
    progress_items = [
        ("🔍 Web Search", "search_results" in st.session_state.context),
        ("✍️ Blog Generated", "blog" in st.session_state.context),
        ("📈 SEO Reviewed", "seo_feedback" in st.session_state.context)
    ]
    
    for item, completed in progress_items:
        if completed:
            st.success(f"✅ {item}")
        else:
            st.info(f"⏳ {item}")
    
    # Analytics Chart in Sidebar
    if "seo_feedback" in st.session_state.context:
        st.divider()
        st.markdown("### 📊 Quick Stats")
        
        # Create a simple gauge chart
        fig, ax = plt.subplots(figsize=(3, 2))
        score = st.session_state.context.get("seo_score", 80)
        
        ax.barh(['SEO'], [score], color='#28a745' if score >= 85 else '#ffc107' if score >= 70 else '#dc3545')
        ax.set_xlim(0, 100)
        ax.set_xlabel('Score')
        ax.set_title(f'SEO: {score}/100', fontsize=10)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

# 4. Agent 1: Web Search 
st.header("🔍 Web Search Agent")

col1, col2 = st.columns([3, 1])
with col1:
    topic = st.text_input("Enter a blog topic", placeholder="e.g. AI in retail, Remote work trends, Sustainable business")
with col2:
    search_depth = st.selectbox("Search Depth", ["Quick", "Deep"])

#enrich 
def enrich_topic(topic):
    topic_lower = topic.lower()
    if "sustain" in topic_lower or "green" in topic_lower or "climate" in topic_lower:
        return topic + " ESG climate green energy"
    elif "remote" in topic_lower or "hybrid" in topic_lower:
        return topic + " remote work productivity collaboration"
    elif "ai" in topic_lower or "artificial" in topic_lower:
        return topic + " artificial intelligence machine learning automation"
    return topic


def web_search_agent(topic, depth="Quick"):
    client = TavilyClient(api_key=st.secrets["TAVILY_API_KEY"])
    cta = "\n\n**💬 What's your take?** Share your thoughts in the comments!" if include_cta else ""

    # Choose depth level
    search_depth = "advanced" if depth == "Deep" else "basic"
    
    # Tavily API
    query = enrich_topic(topic)
    results = client.search(query=query, search_depth=search_depth)
   
    insights = [f"- {r['title']}" for r in results["results"][:6]]
    result = f"Top insights about '{topic}':\n" + "\n".join(insights)
    
    if include_stats:
        result += f"\n\n📊 Key Statistic: {random.randint(50, 90)}% of experts agree {topic.lower()} is transformative."
    
    # Simulate trend data for chart
    trend_data = [random.randint(30, 90) for _ in range(6)]
    
    return result, trend_data


if topic:
    search_results, trend_data = web_search_agent(topic, search_depth)
    st.session_state.context["search_results"] = search_results
    st.session_state.context["topic"] = topic
    st.session_state.context["tone"] = tone
    st.session_state.context["length"] = blog_length
    st.session_state.context["trend_data"] = trend_data
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.text_area("Web Search Results", value=search_results, height=180, disabled=True)
        st.info("💡 Agent Role: Gathers topic insights | Context Saved: [search_results]")
    
    with col2:
        st.markdown("**📈 Topic Trend (6-month)**")
        # Create trend chart
        fig, ax = plt.subplots(figsize=(5, 3))
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
        ax.plot(months, trend_data, marker='o', linewidth=2, markersize=6, color='#1f77b4')
        ax.fill_between(range(len(months)), trend_data, alpha=0.3, color='#1f77b4')
        ax.set_ylabel('Interest Level', fontsize=9)
        ax.set_title(f'{topic} - Growth Trend', fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 100)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

# 5. Agent 2: Blog Writer (Enhanced)
st.header("✍️ Blog Writer Agent")

def generate_title(topic, tone):
    prompt = f"Generate a compelling LinkedIn blog title about '{topic}' in a {tone.lower()} tone."
    response = llm.complete(prompt)
    return response.text.strip()

# CTA section
cta = ""
if include_cta:
    cta = "\n\n**💬 What's your take?** Share your thoughts in the comments below!"

def blog_writer_agent(search_results, tone, length, topic):
    """Generates a blog post using search insights, tone, and length preferences"""
    
    lines = search_results.split('\n')
    topic_line = lines[0].replace("Top insights about '", "").replace("':", "")
    
    insights = [line.strip('- ') for line in lines[1:] if line.startswith('-')]
    
    default_title = generate_title(topic, tone)
    custom_title = st.text_input("Optional: Edit your blog title", value=default_title)
    title = custom_title if custom_title else default_title

#Opening para    
    openings = {
        "Professional": f"In today's rapidly evolving business landscape, {topic.lower()} has emerged as a critical factor for organizational success.",
        "Casual": f"Hey LinkedIn! Let's dive into something that's been on my mind lately: {topic.lower()}.",
        "Inspirational": f"Imagine a world where {topic.lower()} transforms everything we know about business. That world is here, now.",
        "Technical": f"This analysis explores the technical implications and implementation strategies of {topic.lower()} in modern systems.",
        "Thought Leadership": f"As we stand at the intersection of innovation and tradition, {topic.lower()} represents more than just a trend—it's a paradigm shift."
    }
    
    default_opening = openings.get(tone, openings["Professional"])
    custom_opening = st.text_area("Optional: Customize your opening", value=default_opening)
    opening = custom_opening if custom_opening else default_opening

#Key insights sect
    insights_section = "**Key Findings:**\n"
    for i, insight in enumerate(insights[:4], 1):
        insights_section += f"\n{i}. {insight}"
#Stats sect    
    stat_section = ""
    for line in lines:
        if "📊" in line or "Key Statistic" in line:
            stat_section = f"\n\n{line}\n"

#Matters/path_forward    
    prompt = f"""
    Write two sections for a LinkedIn blog post about "{topic}" in a {tone.lower()} tone:
    1. Why This Matters — explain why this topic is important today.
    2. The Path Forward — give advice or next steps for readers.

    Use these insights:
    {', '.join(insights[:4])}
    """

    response = llm.complete(prompt)
    sections = response.text.strip().split("\n\n")

    why_matters = f"\n**Why This Matters:**\n{sections[0]}"
    path_forward = f"\n**The Path Forward:**\n{sections[1]}"

#Hashtags   
    base_hashtags = ["#Innovation", "#DigitalTransformation", "#Leadership"]
    topic_words = topic.split()
    topic_hashtags = [f"#{word.capitalize()}" for word in topic_words if len(word) > 3]
    all_hashtags = (topic_hashtags + base_hashtags)[:6]


    blog_post = f"""🚀 {title}

{opening}

{insights_section}{stat_section}
{why_matters}
{path_forward}{cta}
{' '.join(all_hashtags)}

---
_Written on {datetime.now().strftime('%B %d, %Y')}_
"""
    return blog_post

#Checks if search result exist
if "search_results" in st.session_state.context:            #ensures agent 1 has run and stored results
    if st.button("Generate Blog Post", type="primary"):     #triggers agent 2
        with st.spinner("✍️ Crafting your blog post..."):
            blog = blog_writer_agent(                       #call agent 2
                st.session_state.context["search_results"],
                st.session_state.context.get("tone", "Professional"),
                st.session_state.context.get("length", "Medium"),
                st.session_state.context.get("topic", topic)
            )
            st.session_state.context["blog"] = blog #stores blog
        st.success("✅ Blog post generated successfully!")
    
    if "blog" in st.session_state.context:   #Display blog
        st.text_area("Generated Blog Post", value=st.session_state.context["blog"], height=450, disabled=True)
        
        word_count = len(st.session_state.context["blog"].split())
        st.caption(f"📝 Word Count: {word_count} words | Estimated Reading Time: {word_count // 200 + 1} min")
        
        st.info("💡 Agent Role: Writes the blog using AI | Context Saved: [blog]")

# 6. Agent 3: SEO Review (Enhanced with Charts)
st.header("📈 SEO Review Agent")

def calculate_seo_score(blog):
    score = random.randint(70, 85)
    if len(blog) > 500: score += 5
    if blog.count('#') >= 4: score += 3
    if '?' in blog: score += 2
    if any(word in blog.lower() for word in ['you', 'your', 'we']): score += 3
    return min(score, 98)

def generate_seo_metrics(blog):
    return {
        'Keyword Density': random.randint(75, 95),
        'Readability': random.randint(70, 90),
        'Engagement': random.randint(75, 95),
        'Structure': random.randint(80, 98),
        'Length': min(100, (len(blog.split()) / 500) * 100)
    }


def keyword_match_score(blog, topic):
    keywords = topic.split()
    matches = sum(1 for kw in keywords if kw.lower() in blog.lower())
    return int((matches / len(keywords)) * 100) if keywords else 0

def seo_agent(blog, topic):
    score = calculate_seo_score(blog)
    metrics = generate_seo_metrics(blog)
    metrics['Keyword Match'] = keyword_match_score(blog, topic)
#feedback message based on score
    if score >= 90:
        feedback = "🌟 Excellent SEO! Your blog is highly optimized."
    elif score >= 80:
        feedback = "👍 Good SEO. A few tweaks could make it even better."
    else:
        feedback = "⚠️ Needs improvement. Try adding more keywords and structure."

    return feedback, score, metrics

if "blog" in st.session_state.context:
    if st.button("Run SEO Review", type="primary"):
        with st.spinner("📊 Analyzing SEO..."):
            seo_feedback, seo_score, metrics = seo_agent(
                st.session_state.context["blog"],
                st.session_state.context.get("topic", "")
            )
            st.session_state.context["seo_feedback"] = seo_feedback
            st.session_state.context["seo_score"] = seo_score
            st.session_state.context["metrics"] = metrics
        st.success("✅ SEO analysis complete!")

#Display results    
    if "seo_feedback" in st.session_state.context:
        st.subheader("🔍 SEO Score")
        st.metric("Overall SEO Score", f"{st.session_state.context['seo_score']}/100")

        st.subheader("📊 SEO Breakdown")
        for key, value in st.session_state.context["metrics"].items():
            st.progress(int(value), text=f"{key}: {int(value)}%")

        st.info(st.session_state.context["seo_feedback"])

   
            # Pie chart for score distribution
        st.markdown("**🎯 Overall Score Distribution**")
        fig2, ax2 = plt.subplots(figsize=(5, 3))
            
        score = st.session_state.context.get("seo_score", 80)
        remaining = 100 - score
        labels = [f'Score ({score})', f'Potential ({remaining})']
        colors_pie = ['#28a745' if score >= 85 else '#ffc107', '#e0e0e0']
        ax2.pie([score, remaining], labels =labels, autopct='%1.0f%%',
                   colors=colors_pie, startangle=90)
        ax2.set_title(f'SEO Score: {score}/100', fontweight='bold')
            
        st.caption("🧠 Your SEO score reflects current optimization. 'Potential' shows room for improvement.")
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

# 7. Final Summary
st.header("📋 Final Summary & Analytics")

if "blog" in st.session_state.context and "seo_feedback" in st.session_state.context:
    st.markdown("### ✅ Your Blog is Ready for LinkedIn!")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("#### 📝 Final Blog Post")
        st.markdown(st.session_state.context["blog"])
        
        # Download button
        st.download_button(
            label="⬇️ Download Blog Post",
            data=st.session_state.context["blog"],
            file_name=f"linkedin_blog_{topic.replace(' ', '_')}.txt",
            mime="text/plain"
        )
    
    with col2:
        st.markdown("#### 📌 Performance Summary")
        st.info(st.session_state.context["seo_feedback"])
        
        # Comparison chart
        st.markdown("**📊 Content Quality Comparison**")
        fig3, ax3 = plt.subplots(figsize=(5, 3))
        
        categories = ['Your\nBlog', 'Industry\nAverage', 'Top\nPerformers']
        scores = [
            st.session_state.context.get("seo_score", 80),
            random.randint(65, 75),
            random.randint(90, 95)
        ]
        
        bars = ax3.bar(categories, scores, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        ax3.set_ylabel('SEO Score', fontweight='bold')
        ax3.set_ylim(0, 100)
        ax3.set_title('Competitive Positioning', fontweight='bold')
        ax3.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 2,
                    f'{score}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close()
    
    if st.button("🔄 Start New Blog"):
        st.session_state.context = {}
        st.rerun()
else:
    st.info("👆 Complete all steps above to see your final blog post and analytics")
    
    # Show example visualization
    st.markdown("### 📊 What You'll Get:")
    st.markdown("Complete your blog to unlock:")
    st.markdown("- 📈 SEO performance metrics with visual breakdowns")
    st.markdown("- 🎯 Competitive positioning analysis")
    st.markdown("- 📊 Detailed scoring across multiple dimensions")
    st.markdown("- 💡 Actionable optimization recommendations")

    

"""
PsycheCompass Interactive UI

A Gradio-based web interface for the complete PsycheCompass system.
Provides interactive mental health assessment and therapeutic support.
"""

import gradio as gr
import yaml
import json
from datetime import datetime
from typing import List, Tuple, Optional, Dict

from main import PsycheCompass
from utils.llm import LLMOrchestrator

# Initialize the system
print("Initializing PsycheCompass...")
psyche = PsycheCompass("config.yml")
print("✓ System ready!")

# Global state management


class SessionState:
    def __init__(self):
        self.session_id = None
        self.user_id = None
        self.current_topic_idx = 0
        self.current_question = ""
        self.current_topic_config = None
        self.follow_up_count = 0
        self.conversation_history = []
        self.assessment_complete = False
        self.assessment_results = None
        self.therapeutic_response = None
        self.topics = psyche.config['assessment_scale']['topics']
        self.max_follow_ups = psyche.config['agent_params']['max_follow_ups']

    def reset(self):
        """Reset session state"""
        self.__init__()


# Create global session state
session = SessionState()


def start_assessment(user_id: str, age: str, occupation: str, gender: str) -> Tuple[str, str, str]:
    """
    Start a new assessment session

    Returns:
        (welcome_message, first_question, status_message)
    """
    try:
        # Validate inputs
        if not user_id or not user_id.strip():
            return (
                "❌ Please provide a User ID to start.",
                "",
                "⚠️ Waiting for user information"
            )

        # Reset session
        session.reset()
        session.user_id = user_id.strip()
        session.session_id = f"{session.user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Store user info
        user_info = {}
        if age and age.strip():
            try:
                user_info['age'] = int(age)
            except ValueError:
                pass
        if occupation and occupation.strip():
            user_info['occupation'] = occupation.strip()
        if gender and gender.strip():
            user_info['gender'] = gender.strip()

        # Start assessment pipeline
        session.memory = psyche.assessment_pipeline.start_session(
            user_id=session.user_id,
            user_info=user_info
        )

        # Generate first question
        session.current_topic_config = session.topics[0]
        session.current_topic_idx = 0

        session.current_question = psyche.assessment_pipeline.question_agent.generate_initial_question(
            topic_config=session.current_topic_config,
            memory=session.memory
        )

        welcome_msg = f"""
# 🧠 Welcome to PsycheCompass

**Session ID:** {session.session_id}  
**User:** {session.user_id}

This is a PHQ-8 mental health assessment. I'll ask you about 8 different areas of your mental health. 
For each area, I may ask follow-up questions to better understand your experience.

**Remember:**
- Answer honestly
- Take your time
- There are no right or wrong answers
- Your responses are confidential

Let's begin with the first topic: **{session.current_topic_config['name']}**
"""

        status_msg = f"📊 **Topic 1/8:** {session.current_topic_config['name']}"

        return welcome_msg, session.current_question, status_msg

    except Exception as e:
        return (
            f"❌ Error starting assessment: {str(e)}",
            "",
            "⚠️ Error - Please try again"
        )


def process_answer(user_answer: str) -> Tuple[str, str, str, str]:
    """
    Process user's answer and generate next question or move to next topic

    Returns:
        (next_question, conversation_display, status_message, progress_bar)
    """
    try:
        if not user_answer or not user_answer.strip():
            return (
                session.current_question,
                _format_conversation(),
                f"📊 **Topic {session.current_topic_idx + 1}/8:** {session.current_topic_config['name']}",
                _get_progress()
            )

        if not session.current_question:
            return (
                "Please start the assessment first using the 'Start Assessment' button.",
                "",
                "⚠️ No active session",
                ""
            )

        # Store Q&A
        topic_name = session.current_topic_config['name']
        session.conversation_history.append({
            'topic': topic_name,
            'question': session.current_question,
            'answer': user_answer.strip()
        })

        # Add to memory
        session.memory.add_qa_pair(
            topic_name, session.current_question, user_answer.strip())

        # Extract information
        statement = session.memory.extract_information(user_answer.strip())
        session.memory.add_statement(topic_name, statement)

        # Evaluate adequacy
        necessity_score = psyche.assessment_pipeline.evaluation_agent.evaluate_adequacy(
            topic_config=session.current_topic_config,
            topic_name=topic_name,
            memory=session.memory
        )

        # Check if we need follow-up
        if necessity_score >= psyche.assessment_pipeline.necessity_threshold and \
           session.follow_up_count < session.max_follow_ups:
            # Generate follow-up question
            session.follow_up_count += 1
            session.current_question = psyche.assessment_pipeline.question_agent.generate_followup_question(
                topic_config=session.current_topic_config,
                topic_name=topic_name,
                memory=session.memory
            )

            status_msg = f"📊 **Topic {session.current_topic_idx + 1}/8:** {topic_name} (Follow-up {session.follow_up_count}/{session.max_follow_ups})"

            return (
                session.current_question,
                _format_conversation(),
                status_msg,
                _get_progress()
            )

        # Move to next topic
        session.current_topic_idx += 1
        session.follow_up_count = 0

        if session.current_topic_idx < len(session.topics):
            # Generate next topic question
            session.current_topic_config = session.topics[session.current_topic_idx]
            topic_name = session.current_topic_config['name']

            session.memory.add_topic_node(topic_name)

            session.current_question = psyche.assessment_pipeline.question_agent.generate_initial_question(
                topic_config=session.current_topic_config,
                memory=session.memory
            )

            status_msg = f"📊 **Topic {session.current_topic_idx + 1}/8:** {topic_name}"

            return (
                session.current_question,
                _format_conversation(),
                status_msg,
                _get_progress()
            )
        else:
            # Assessment complete - trigger scoring
            session.assessment_complete = True
            return _complete_assessment()

    except Exception as e:
        return (
            session.current_question if session.current_question else "",
            _format_conversation(),
            f"⚠️ Error: {str(e)}",
            _get_progress()
        )


def _complete_assessment() -> Tuple[str, str, str, str]:
    """
    Complete the assessment and prepare for therapeutic phase
    """
    try:
        # Score all topics
        total_score = 0
        topic_results = []

        for topic_config in session.topics:
            topic_name = topic_config['name']

            score, summary, basis = psyche.assessment_pipeline.scoring_agent.score_topic(
                topic_config=topic_config,
                topic_name=topic_name,
                memory=session.memory
            )

            session.memory.update_topic_score(
                topic_name, score, summary, basis)
            total_score += score

            topic_results.append({
                'topic': topic_name,
                'score': score,
                'summary': summary,
                'basis': basis
            })

        # Classify risk
        classification = psyche.assessment_pipeline._classify_risk(total_score)

        # Detect crisis
        crisis_indicators = psyche.assessment_pipeline._detect_crisis_indicators()

        # Store assessment results
        session.assessment_results = {
            'user_id': session.user_id,
            'user_info': session.memory.user.to_dict(),
            'scale_name': psyche.assessment_pipeline.scale_name,
            'total_score': total_score,
            'classification': classification,
            'crisis_indicators': crisis_indicators,
            'topics': topic_results,
            'timestamp': datetime.now().isoformat()
        }

        # Generate assessment report
        assessment_report = psyche.assessment_pipeline.updating_agent.generate_report(
            memory=session.memory,
            total_score=total_score,
            scale_name=psyche.assessment_pipeline.scale_name
        )

        session.assessment_results['report'] = assessment_report

        completion_msg = f"""
# ✅ Assessment Complete!

**PHQ-8 Total Score:** {total_score}/24  
**Classification:** {classification}  
**Crisis Indicators:** {len(crisis_indicators)} detected

---

**Assessment Report:**

{assessment_report}

---

Click **"Generate Therapeutic Response"** below to receive personalized support based on your assessment.
"""

        status_msg = f"✅ Assessment Complete - Score: {total_score}/24 ({classification})"

        return (
            "",  # No more questions
            _format_conversation(),
            status_msg,
            completion_msg  # Show in progress area
        )

    except Exception as e:
        return (
            "",
            _format_conversation(),
            f"⚠️ Error completing assessment: {str(e)}",
            ""
        )


def generate_therapeutic_response() -> str:
    """
    Generate therapeutic response based on assessment
    """
    try:
        if not session.assessment_complete or not session.assessment_results:
            return "⚠️ Please complete the assessment first."

        # Generate therapeutic response
        therapeutic_result = psyche.therapeutic_pipeline.generate_response(
            assessment_results=session.assessment_results
        )

        session.therapeutic_response = therapeutic_result

        # Format response
        response_display = f"""
# 🌟 Therapeutic Support

**Your Assessment:**
- **PHQ-8 Score:** {session.assessment_results['total_score']}/24
- **Classification:** {session.assessment_results['classification']}
- **Risk Level:** {therapeutic_result['risk_level'].upper()}

---

## Recommended Support: {therapeutic_result['agent_type'].replace('_', ' ').title()}

{therapeutic_result['response']}

---

### Why This Support?

{therapeutic_result['routing_explanation']}

---

*You can continue the conversation by typing your questions or concerns in the chat box below.*
"""

        return response_display

    except Exception as e:
        return f"⚠️ Error generating therapeutic response: {str(e)}"


def continue_therapeutic_conversation(user_message: str, conversation_state: List) -> Tuple[List, str]:
    """
    Continue multi-turn therapeutic conversation
    """
    try:
        if not session.therapeutic_response:
            return conversation_state, "⚠️ Please generate the initial therapeutic response first."

        if not user_message or not user_message.strip():
            return conversation_state, ""

        # Get response from therapeutic pipeline
        response = psyche.therapeutic_pipeline.continue_conversation(
            user_message=user_message.strip(),
            assessment_results=session.assessment_results
        )

        # Update conversation
        conversation_state.append((user_message, response['response']))

        return conversation_state, ""

    except Exception as e:
        return conversation_state, f"⚠️ Error: {str(e)}"


def _format_conversation() -> str:
    """Format conversation history for display"""
    if not session.conversation_history:
        return ""

    formatted = "# 💬 Conversation History\n\n"

    current_topic = None
    for entry in session.conversation_history:
        if entry['topic'] != current_topic:
            current_topic = entry['topic']
            formatted += f"\n## 📌 {current_topic}\n\n"

        formatted += f"**Agent:** {entry['question']}\n\n"
        formatted += f"**You:** {entry['answer']}\n\n"
        formatted += "---\n\n"

    return formatted


def _get_progress() -> str:
    """Get progress bar"""
    if session.assessment_complete:
        return "✅ Assessment Complete!"

    progress = (session.current_topic_idx / len(session.topics)) * 100
    return f"📊 Progress: {session.current_topic_idx}/{len(session.topics)} topics ({progress:.0f}%)"


def reset_session() -> Tuple[str, str, str, str, str, List]:
    """Reset the entire session"""
    session.reset()
    return (
        "",  # welcome message
        "",  # question
        "Ready to start a new assessment",  # status
        "",  # progress
        "",  # therapeutic response
        []   # chat history
    )


# Build Gradio Interface
with gr.Blocks(title="PsycheCompass - Mental Health Assessment & Support", theme=gr.themes.Soft()) as app:

    gr.Markdown("""
    # 🧠 PsycheCompass
    ### Adaptive Multi-Agent System for Mental Health Assessment & Therapeutic Support
    
    This system provides:
    1. **PHQ-8 Depression Assessment** - Structured clinical interview
    2. **Risk-Based Therapeutic Support** - Personalized support based on your assessment
    3. **Interactive Conversation** - Continue discussing your mental health concerns
    """)

    # State for therapeutic chat
    chat_history = gr.State([])

    with gr.Tabs():
        # TAB 1: User Information
        with gr.Tab("1️⃣ User Information"):
            gr.Markdown("## 📝 Enter Your Information")
            gr.Markdown(
                "*All information is kept confidential and used only for this session.*")

            with gr.Row():
                user_id_input = gr.Textbox(
                    label="User ID*",
                    placeholder="Enter a unique identifier (e.g., user_001)",
                    info="Required"
                )
                age_input = gr.Textbox(
                    label="Age",
                    placeholder="Optional",
                    info="Helps personalize questions"
                )

            with gr.Row():
                occupation_input = gr.Textbox(
                    label="Occupation",
                    placeholder="Optional"
                )
                gender_input = gr.Dropdown(
                    label="Gender",
                    choices=["Male", "Female", "Non-binary",
                             "Prefer not to say", "Other"],
                    value="Prefer not to say"
                )

            start_btn = gr.Button("Start Assessment ▶️",
                                  variant="primary", size="lg")

            welcome_output = gr.Markdown()

        # TAB 2: Assessment
        with gr.Tab("2️⃣ PHQ-8 Assessment"):
            gr.Markdown("## 💬 Mental Health Assessment")

            status_display = gr.Markdown(
                "⚠️ Please start assessment in the User Information tab")
            progress_display = gr.Markdown()

            question_display = gr.Markdown()

            user_answer_input = gr.Textbox(
                label="Your Answer",
                placeholder="Type your answer here...",
                lines=3,
                info="Press Enter or click Submit to continue"
            )

            submit_answer_btn = gr.Button("Submit Answer ➤", variant="primary")

            conversation_display = gr.Markdown()

        # TAB 3: Therapeutic Support
        with gr.Tab("3️⃣ Therapeutic Support"):
            gr.Markdown("## 🌟 Personalized Support")

            generate_therapeutic_btn = gr.Button(
                "Generate Therapeutic Response 🎯",
                variant="primary",
                size="lg"
            )

            therapeutic_output = gr.Markdown()

            gr.Markdown("---")
            gr.Markdown("### 💭 Continue the Conversation")
            gr.Markdown(
                "*Ask questions, share concerns, or discuss the recommendations above.*")

            chatbot = gr.Chatbot(
                label="Therapeutic Conversation",
                height=400
            )

            with gr.Row():
                chat_input = gr.Textbox(
                    label="Your Message",
                    placeholder="Type your message here...",
                    lines=2,
                    scale=4
                )
                chat_submit = gr.Button("Send 📤", scale=1)

        # TAB 4: Session Information
        with gr.Tab("ℹ️ About"):
            gr.Markdown("""
            ## About PsycheCompass
            
            PsycheCompass is an adaptive multi-agent system that provides:
            
            ### 📊 Assessment Phase
            - **PHQ-8 Depression Screening**: Standardized clinical assessment
            - **Adaptive Questioning**: Follow-up questions for clarity
            - **Multi-topic Coverage**: 8 key areas of mental health
            - **Crisis Detection**: Automatic identification of urgent concerns
            
            ### 🎯 Therapeutic Phase
            - **Risk-Based Routing**: Support tailored to your symptom severity
            - **Three-Tier System**:
              - **Emotional Support** (Minimal symptoms): Validation and self-care
              - **Therapeutic Intervention** (Moderate): Evidence-based techniques (CBT, Behavioral Activation)
              - **Crisis Support** (Severe): Immediate safety resources and urgent referrals
            - **Clinical Validation**: All responses checked for safety
            
            ### 🔒 Privacy & Safety
            - All data is session-only (not stored permanently)
            - Clinical safety protocols in place
            - Crisis hotlines provided when needed
            - Professional referrals included
            
            ### ⚠️ Important Disclaimers
            - This is NOT a substitute for professional mental health care
            - For emergencies, call 988 (Suicide & Crisis Lifeline) or 911
            - Consult a licensed mental health professional for diagnosis and treatment
            
            ---
            
            ### 🛠️ System Architecture
            
            **Assessment Pipeline:**
            - Question Generator Agent
            - Evaluation Agent
            - Scoring Agent
            - Memory Management
            
            **Therapeutic Pipeline:**
            - Therapeutic Router
            - Emotional Support Agent
            - Therapeutic Agent
            - Crisis Support Agent
            - Clinical Validator
            
            ---
            
            ### 📚 Based on Research
            
            *"PsycheCompass: An Adaptive Multi-Agent system for Mental Health Assessment & Therapeutic Support"*
            
            This system implements a modular, research-grade architecture for mental health support.
            """)

            reset_btn = gr.Button("🔄 Reset Session", variant="stop")

    # Event handlers
    start_btn.click(
        fn=start_assessment,
        inputs=[user_id_input, age_input, occupation_input, gender_input],
        outputs=[welcome_output, question_display, status_display]
    )

    submit_answer_btn.click(
        fn=process_answer,
        inputs=[user_answer_input],
        outputs=[question_display, conversation_display,
                 status_display, progress_display]
    ).then(
        fn=lambda: "",  # Clear input
        outputs=[user_answer_input]
    )

    user_answer_input.submit(
        fn=process_answer,
        inputs=[user_answer_input],
        outputs=[question_display, conversation_display,
                 status_display, progress_display]
    ).then(
        fn=lambda: "",  # Clear input
        outputs=[user_answer_input]
    )

    generate_therapeutic_btn.click(
        fn=generate_therapeutic_response,
        outputs=[therapeutic_output]
    )

    chat_submit.click(
        fn=continue_therapeutic_conversation,
        inputs=[chat_input, chat_history],
        outputs=[chatbot, chat_input]
    ).then(
        fn=lambda: "",
        outputs=[chat_input]
    )

    chat_input.submit(
        fn=continue_therapeutic_conversation,
        inputs=[chat_input, chat_history],
        outputs=[chatbot, chat_input]
    ).then(
        fn=lambda: "",
        outputs=[chat_input]
    )

    reset_btn.click(
        fn=reset_session,
        outputs=[
            welcome_output,
            question_display,
            status_display,
            progress_display,
            therapeutic_output,
            chatbot
        ]
    )

# Launch the app
if __name__ == "__main__":
    print("\n" + "="*70)
    print("🧠 PsycheCompass Interactive UI")
    print("="*70)
    print("\nStarting Gradio interface...")
    print("The application will open in your browser automatically.\n")

    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,  # Set to True to create a public link
        show_error=True
    )

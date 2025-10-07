import streamlit as st
import pandas as pd
from utils import load_model_stuff, generate_road, predict_risk  # Backend imports

# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(page_title="Pick the Safer Road", page_icon="🚗", layout="wide")

# Load backend once
model, scaler, feature_names = load_model_stuff()

# Enhanced road images
road_images = {
    'urban': 'images/urban-road.jpg',
    'rural': 'images/rural-road.jpg',
    'highway': 'images/highway-road.jpg',
    ('rural', 'night', 'clear', 2): 'images/night-country-road.jpg',
    ('urban', 'daylight', 'rainy', 2): 'images/Wet-city-street-day.jpg',
    ('highway', 'dim', 'foggy', 4): 'images/foggy-highway.jpg',
}

def get_road_image(road_type, lighting, weather, num_lanes):
    key = (road_type, lighting, weather, num_lanes)
    return road_images.get(key, road_images.get(road_type, road_images['rural']))

# -----------------------------
# CSS for styling
# -----------------------------
st.markdown("""
<style>
    .main { background: linear-gradient(to bottom, #e3f2fd, #f3e5f5); padding: 20px; }
    .stButton > button { background-color: #2e7d32; color: white; border: none; border-radius: 10px; padding: 12px 24px; font-size: 16px; box-shadow: 0 2px 4px rgba(0,0,0,0.2); }
    .stButton > button:hover { background-color: #4caf50; transform: translateY(-1px); }
    .stRadio > div > label { font-size: 18px; font-weight: bold; color: #333; }
    h1 { color: #1b5e20; text-align: center; font-family: 'Georgia', serif; }
    .metric-card { background: white; padding: 15px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); text-align: center; }
    .road-table { border: 2px solid #2e7d32; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
    .road-table th { background-color: #4caf50; color: white; border: 1px solid #2e7d32; padding: 8px; text-align: left; }
    .road-table td { border: 1px solid #2e7d32; padding: 6px; background-color: #f9f9f9; }
    .road-table tr:nth-child(even) td { background-color: #e8f5e8; }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Header
# -----------------------------
st.title("🚗 Pick the Safer Road Game")
st.markdown("Guess which road's less accident-prone. Model reveals the risks after your pick. Score high! (5 rounds total)")

# -----------------------------
# Session state initialization
# -----------------------------
if 'score' not in st.session_state:
    st.session_state.update({
        'score': 0,
        'total_questions': 0,
        'game_active': True,
        'game_over': False,
        'road1': None,
        'road2': None,
        'risk1': None,
        'risk2': None,
        'user_choice': None,
        'max_rounds': 5
    })

# -----------------------------
# Generate roads for the round
# -----------------------------
if st.session_state.game_active and not st.session_state.game_over:
    seed1 = st.session_state.total_questions * 2
    seed2 = st.session_state.total_questions * 2 + 1
    road1 = generate_road(seed=seed1)
    road2 = generate_road(seed=seed2)
    
    while road1 == road2:
        road2 = generate_road(seed=seed2 + 1)
    
    df1 = pd.DataFrame([road1])
    df2 = pd.DataFrame([road2])
    
    risk1 = predict_risk(df1, model, feature_names, scaler)
    risk2 = predict_risk(df2, model, feature_names, scaler)
    
    st.session_state.update({
        'road1': road1,
        'road2': road2,
        'risk1': risk1,
        'risk2': risk2
    })

# -----------------------------
# Display roads
# -----------------------------
if st.session_state.game_active and not st.session_state.game_over:
    col1, col2 = st.columns(2)
    
    def display_road(col, road, risk, title):
        with col:
            st.subheader(f"🛣️ {title}")
            img_url = get_road_image(road['road_type'], road['lighting'], road['weather'], road['num_lanes'])
            caption = f"{road['road_type'].title()} road: {road['lighting']} {road['weather']} weather, {road['num_lanes']} lanes"
            st.image(img_url, width=500, caption=caption)
            
            features = list(road.items())
            top_features = pd.DataFrame(features[:6], columns=['Feature', 'Value'])
            bottom_features = pd.DataFrame(features[6:], columns=['Feature', 'Value'])
            
            st.markdown("**Top Conditions:**")
            st.dataframe(top_features, width='stretch', height=250)
            st.markdown("**Other Details:**")
            st.dataframe(bottom_features, width='stretch', height=250)
    
    display_road(col1, st.session_state.road1, st.session_state.risk1, "Road 1")
    display_road(col2, st.session_state.road2, st.session_state.risk2, "Road 2")
    
    st.markdown("---")
    user_choice = st.radio("Safer road? (Lower risk)", ['Road 1', 'Road 2'])
    
    col_submit, col_reset = st.columns([3, 1])
    with col_submit:
        if st.button("🚦 Submit Guess", type="primary"):
            st.session_state.game_active = False
            st.session_state.user_choice = user_choice
            safer = 'Road 1' if st.session_state.risk1 < st.session_state.risk2 else 'Road 2'
            if user_choice == safer:
                if st.session_state.total_questions < st.session_state.max_rounds:
                    st.session_state.score += 1
                st.balloons()
            
            # Increment total_questions but clamp to max_rounds
            st.session_state.total_questions = min(st.session_state.total_questions + 1, st.session_state.max_rounds)
    
    with col_reset:
        if st.button("🔄 New Game"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

# -----------------------------
# Results / Verdict
# -----------------------------
if not st.session_state.game_active and not st.session_state.game_over:
    st.markdown("---")
    st.subheader("📊 Verdict")
    
    col_r1, col_r2 = st.columns(2)
    with col_r1:
        st.metric("Road 1 Risk", f"{st.session_state.risk1:.3f}")
    with col_r2:
        st.metric("Road 2 Risk", f"{st.session_state.risk2:.3f}")
    
    safer = 'Road 1' if st.session_state.risk1 < st.session_state.risk2 else 'Road 2'
    st.info(f"Safer: **{safer}**.")
    
    if st.session_state.user_choice == safer:
        st.success("🎉 Nailed it!")
    else:
        st.warning("😅 Nope.")
    
    score_pct = (st.session_state.score / st.session_state.max_rounds) * 100
    st.metric("Round Score", f"{st.session_state.score}/{st.session_state.total_questions}", f"{score_pct:.0f}%")
    
    if st.session_state.total_questions >= st.session_state.max_rounds:
        st.session_state.game_over = True
        st.success(f"🎊 Game Over! Final Score: {st.session_state.score}/{st.session_state.max_rounds} ({score_pct:.0f}%)")
        if st.button("🔄 Play Again"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
    else:
        if st.button("➡️ Next Round"):
            st.session_state.game_active = True
            st.rerun()

# -----------------------------
# End screen if game over
# -----------------------------
elif st.session_state.game_over:
    st.markdown("---")
    st.subheader("🏁 Game Complete!")
    final_score = st.session_state.score
    score_pct = min((final_score / st.session_state.max_rounds) * 100, 100)  # clamp 100%
    
    col_score, col_progress = st.columns(2)
    with col_score:
        st.metric("Final Score", f"{final_score}/{st.session_state.max_rounds}", f"{score_pct:.0f}%")
    with col_progress:
        st.progress(score_pct / 100)
    
    if st.button("🔄 Play Again"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# -------------------------------
# Footer
# -------------------------------
st.write("---")
st.write(
    "Built with ❤️ for Kaggle Playground S5E10 | Drive safe! 🚫  \n"
    "[Kaggle](https://www.kaggle.com/sdeogade) | "
    "[LinkedIn](https://www.linkedin.com/in/sahil-deogade-96a9a927b/)"
)

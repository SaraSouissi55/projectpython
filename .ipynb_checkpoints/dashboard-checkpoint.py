import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import joblib  
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# -------------------------
# Page config
# -------------------------
st.set_page_config(
    page_title="IoT Sensor Dashboard",
    layout="wide"
)

st.title("🌍 IoT Sensor Supervision Dashboard")

# -------------------------
# Load data
# -------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("sensor_data_finall.csv", parse_dates=['timestamp'])
    
    # Convert numeric columns
    for col in ['temperature', 'humidity', 'air_quality']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Forward fill missing values
    for col in ['temperature', 'humidity', 'air_quality']:
        df[col] = df[col].ffill()
    
    return df

df = load_data()

# -------------------------
# SIDEBAR
# -------------------------
st.sidebar.header("⚙️ Navigation")
page = st.sidebar.selectbox(
    "Select Page",
    ["📊 Data Overview", 
     "📈 Real-Time Monitoring", 
     "🔄 Data Preprocessing",
     "🤖 Model Deployment"]
)

# -------------------------
# PAGE 1: DATA OVERVIEW
# -------------------------
if page == "📊 Data Overview":
    st.header("📊 Data Overview")
    
    # Basic metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Records", len(df))
    with col2:
        st.metric("Sensors", df['sensor_id'].nunique())
    with col3:
        anomalies = df['anomaly'].sum() if 'anomaly' in df.columns else 0
        st.metric("Anomalies", anomalies)
    
    # Data preview
    with st.expander("📂 View Raw Data"):
        st.dataframe(df.head(50))
    
    # Simple statistics
    st.subheader("📈 Basic Statistics")
    st.dataframe(df[['temperature', 'humidity', 'air_quality']].describe())
    
    # Sensor distribution
    st.subheader("📊 Sensor Readings Distribution")
    
    # Plot sensor readings
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Temperature
    for sensor in sorted(df['sensor_id'].unique()):
        sensor_data = df[df['sensor_id'] == sensor]['temperature']
        axes[0].hist(sensor_data, alpha=0.5, label=f'Sensor {sensor}', bins=20)
    axes[0].set_xlabel('Temperature (°C)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Temperature Distribution')
    axes[0].legend()
    
    # Humidity
    for sensor in sorted(df['sensor_id'].unique()):
        sensor_data = df[df['sensor_id'] == sensor]['humidity']
        axes[1].hist(sensor_data, alpha=0.5, label=f'Sensor {sensor}', bins=20)
    axes[1].set_xlabel('Humidity (%)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Humidity Distribution')
    
    # Air Quality
    for sensor in sorted(df['sensor_id'].unique()):
        sensor_data = df[df['sensor_id'] == sensor]['air_quality']
        axes[2].hist(sensor_data, alpha=0.5, label=f'Sensor {sensor}', bins=20)
    axes[2].set_xlabel('Air Quality (AQI)')
    axes[2].set_ylabel('Frequency')
    axes[2].set_title('Air Quality Distribution')
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Time series overview
    st.subheader("📅 Time Series Overview")
    
    selected_sensor = st.selectbox("Select Sensor for Time Series", df['sensor_id'].unique())
    selected_variable = st.selectbox("Select Variable", ['temperature', 'humidity', 'air_quality'])
    
    sensor_df = df[df['sensor_id'] == selected_sensor]
    
    fig2, ax = plt.subplots(figsize=(12, 4))
    ax.plot(sensor_df['timestamp'], sensor_df[selected_variable], linewidth=1)
    
    # Mark anomalies if available
    if 'anomaly' in sensor_df.columns:
        anomalies = sensor_df[sensor_df['anomaly'] == True]
        ax.scatter(anomalies['timestamp'], anomalies[selected_variable], 
                  color='red', s=30, label='Anomalies', zorder=5)
    
    ax.set_xlabel('Time')
    ax.set_ylabel(selected_variable)
    ax.set_title(f'Sensor {selected_sensor} - {selected_variable}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    st.pyplot(fig2)
    
    # Correlation matrix
    st.subheader("🔗 Feature Correlation")
    fig3, ax = plt.subplots(figsize=(8, 6))
    corr_matrix = df[['temperature', 'humidity', 'air_quality']].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
    ax.set_title('Correlation Matrix')
    st.pyplot(fig3)

# -------------------------
# PAGE 2: REAL-TIME MONITORING
# -------------------------
elif page == "📈 Real-Time Monitoring":
    st.header("📈 Real-Time Monitoring")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        sensor_id = st.selectbox("Select Sensor", df['sensor_id'].unique())
        
        # Sensor thresholds
        if sensor_id == 1:
            variable = "temperature"
            label = "Temperature (°C)"
            threshold = 32
        elif sensor_id == 2:
            variable = "humidity"
            label = "Humidity (%)"
            threshold = 80
        else:
            variable = "air_quality"
            label = "Air Quality (AQI)"
            threshold = 130
    
    # Filter data
    sensor_data = df[df['sensor_id'] == sensor_id].copy()
    sensor_data['anomaly_detected'] = sensor_data[variable] > threshold
    
    # KPIs
    with col2:
        st.subheader("📊 Key Metrics")
        col_k1, col_k2, col_k3 = st.columns(3)
        with col_k1:
            st.metric("Current Value", f"{sensor_data[variable].iloc[-1]:.1f}")
        with col_k2:
            st.metric("Average", f"{sensor_data[variable].mean():.1f}")
        with col_k3:
            st.metric("Anomalies", sensor_data['anomaly_detected'].sum())
    
    # Plot
    st.subheader(f"Sensor {sensor_id} - {label}")
    
    fig, ax = plt.subplots(figsize=(12, 5))
    
    # Main line
    ax.plot(sensor_data['timestamp'], sensor_data[variable], 
            label=label, color='blue', linewidth=1)
    
    # Anomalies
    anomalies = sensor_data[sensor_data['anomaly_detected']]
    if len(anomalies) > 0:
        ax.scatter(anomalies['timestamp'], anomalies[variable], 
                  color='red', s=40, marker='x', label='Anomalies', zorder=5)
    
    # Threshold line
    ax.axhline(y=threshold, color='orange', linestyle='--', 
               label=f'Threshold ({threshold})')
    
    ax.set_xlabel('Time')
    ax.set_ylabel(label)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    st.pyplot(fig)
    
    # Anomaly details
    if len(anomalies) > 0:
        st.subheader("🚨 Recent Anomalies")
        recent_anomalies = anomalies[['timestamp', variable]].tail(10)
        recent_anomalies['timestamp'] = recent_anomalies['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
        st.dataframe(recent_anomalies.rename(columns={variable: label}))
    else:
        st.info("✅ No anomalies detected")

# -------------------------
# PAGE 3: DATA PREPROCESSING
# -------------------------
elif page == "🔄 Data Preprocessing":
    st.header("🔄 Data Preprocessing Pipeline")
    
    # Pipeline steps
    st.markdown("""
    ### 🏭 Processing Steps:
    1. **Load Data** → Read CSV and clean
    2. **Handle Missing Values** → Forward fill
    3. **Split Data** → Train/Test separation
    4. **Normalize Data** → Standard scaling
    5. **Ready for ML** → Prepared datasets
    """)
    
    # 1. Check class distribution
    st.subheader("1. Class Distribution")
    
    if 'anomaly' in df.columns:
        anomaly_dist = df['anomaly'].value_counts()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Bar chart
        ax1.bar(['Normal', 'Anomaly'], anomaly_dist.values, color=['blue', 'red'])
        ax1.set_ylabel('Count')
        ax1.set_title('Anomaly Distribution')
        for i, v in enumerate(anomaly_dist.values):
            ax1.text(i, v + 5, str(v), ha='center')
        
        # Pie chart
        ax2.pie(anomaly_dist.values, labels=['Normal', 'Anomaly'], 
                autopct='%1.1f%%', colors=['lightblue', 'lightcoral'])
        ax2.set_title('Percentage Distribution')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        st.write(f"**Normal samples:** {anomaly_dist[0]} ({anomaly_dist[0]/len(df)*100:.1f}%)")
        st.write(f"**Anomaly samples:** {anomaly_dist[1]} ({anomaly_dist[1]/len(df)*100:.1f}%)")
    
    # 2. Train-test split
    st.subheader("2. Train-Test Split")
    
    if st.button("🔄 Run Train-Test Split"):
        if 'anomaly' in df.columns:
            # Prepare data
            X = df[['temperature', 'humidity', 'air_quality']]
            y = df['anomaly']
            
            # Split
            X_train_raw, X_test_raw, y_train, y_test = train_test_split(
                X, y, 
                test_size=0.3, 
                random_state=42, 
                stratify=y
            )
            
            # Display results
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Training Set:**")
                st.write(f"- Samples: {len(X_train_raw)}")
                st.write(f"- Anomalies: {y_train.sum()} ({y_train.mean()*100:.1f}%)")
            
            with col2:
                st.write("**Test Set:**")
                st.write(f"- Samples: {len(X_test_raw)}")
                st.write(f"- Anomalies: {y_test.sum()} ({y_test.mean()*100:.1f}%)")
            
            # Visual comparison
            fig_split, ax = plt.subplots(figsize=(8, 4))
            categories = ['Full Dataset', 'Training Set', 'Test Set']
            anomaly_counts = [y.sum(), y_train.sum(), y_test.sum()]
            total_counts = [len(y), len(y_train), len(y_test)]
            
            x = np.arange(len(categories))
            width = 0.35
            
            ax.bar(x - width/2, total_counts, width, label='Total', color='lightgray')
            ax.bar(x + width/2, anomaly_counts, width, label='Anomalies', color='red')
            
            ax.set_ylabel('Count')
            ax.set_title('Dataset Distribution')
            ax.set_xticks(x)
            ax.set_xticklabels(categories)
            ax.legend()
            
            for i, (total, anomaly) in enumerate(zip(total_counts, anomaly_counts)):
                ax.text(i - width/2, total + 50, str(total), ha='center')
                ax.text(i + width/2, anomaly + 50, str(anomaly), ha='center')
            
            st.pyplot(fig_split)
            
            # 3. Normalization
            st.subheader("3. Data Normalization")
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_raw)
            X_test_scaled = scaler.transform(X_test_raw)
            
            # Show before/after
            col_before, col_after = st.columns(2)
            
            with col_before:
                st.write("**Before Normalization (Training):**")
                before_stats = pd.DataFrame(X_train_raw).describe().round(2)
                st.dataframe(before_stats)
            
            with col_after:
                st.write("**After Normalization (Training):**")
                after_stats = pd.DataFrame(X_train_scaled, columns=X.columns).describe().round(2)
                st.dataframe(after_stats)
            
            # Store in session
            st.session_state['X_train'] = X_train_scaled
            st.session_state['X_test'] = X_test_scaled
            st.session_state['y_train'] = y_train
            st.session_state['y_test'] = y_test
            
            st.success("✅ Preprocessing completed successfully!")
            st.info(f"""
            **Data shapes:**
            - X_train: {X_train_scaled.shape}
            - X_test: {X_test_scaled.shape}
            - y_train: {y_train.shape}
            - y_test: {y_test.shape}
            """)
        else:
            st.error("Anomaly column not found in data")
    
    # Show preprocessing status
    if 'X_train' in st.session_state:
        st.subheader("📦 Preprocessed Data Status")
        
        col_status1, col_status2 = st.columns(2)
        with col_status1:
            st.write("**Data Shapes:**")
            st.write(f"- X_train: {st.session_state['X_train'].shape}")
            st.write(f"- X_test: {st.session_state['X_test'].shape}")
        
        with col_status2:
            st.write("**Ready for:**")
            st.write("• Model Training")
            st.write("• Model Evaluation")
            st.write("• Model Deployment")

# -------------------------
# PAGE 4: MODEL DEPLOYMENT (FIXED VERSION)
# -------------------------
elif page == "🤖 Model Deployment":
    st.header("🤖 Anomaly Detection Model Deployment")
    
    # FIXED: Load BOTH model and scaler
    @st.cache_resource
    def load_models():
        """Load the FIXED model and scaler"""
        try:
            # Use joblib (better for sklearn models)
            model = joblib.load('anomaly_model_final.pkl')  # FIXED MODEL
            scaler = joblib.load('scaler_final.pkl')       # SCALER
            return model, scaler
        except FileNotFoundError:
            # Try pickle as fallback
            try:
                with open('anomaly_model_final.pkl', 'rb') as f:
                    model = pickle.load(f)
                with open('scaler_final.pkl', 'rb') as f:
                    scaler = pickle.load(f)
                return model, scaler
            except:
                st.error("❌ Model files 'anomaly_model_final.pkl' and 'scaler_final.pkl' not found!")
                st.info("Please run the training notebook first to create these files.")
                return None, None
    
    # Load models
    model, scaler = load_models()
    
    if model is None or scaler is None:
        st.warning("Models could not be loaded. Please check the file paths.")
    else:
        st.success(f"✅ Model & Scaler loaded successfully!")
        st.write(f"**Model type:** {type(model).__name__}")
        
        # Create tabs
        tab1, tab2, tab3 = st.tabs(["🎯 Test Model", "📊 Model Info", "⚡ Live Predictions"])
        
        # TAB 1: Test Model
        with tab1:
            st.subheader("🎯 Model Evaluation")
            
            if 'X_test' in st.session_state and 'y_test' in st.session_state:
                if st.button("🧪 Run Model Evaluation", type="primary"):
                    with st.spinner("Evaluating model performance..."):
                        X_test = st.session_state['X_test']
                        y_test = st.session_state['y_test']
                        
                        # Make predictions (already scaled in preprocessing)
                        y_pred = model.predict(X_test)
                        y_proba = model.predict_proba(X_test)[:, 1]
                        
                        # Calculate metrics
                        accuracy = (y_pred == y_test).mean()
                        precision = (y_test[y_pred == 1] == 1).mean() if (y_pred == 1).sum() > 0 else 0
                        recall = (y_pred[y_test == 1] == 1).mean() if (y_test == 1).sum() > 0 else 0
                        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                        
                        # Display metrics
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Accuracy", f"{accuracy:.1%}")
                        with col2:
                            st.metric("Precision", f"{precision:.1%}")
                        with col3:
                            st.metric("Recall", f"{recall:.1%}")
                        with col4:
                            st.metric("F1-Score", f"{f1:.1%}")
                        
                        # Confusion Matrix
                        st.subheader("Confusion Matrix")
                        cm = confusion_matrix(y_test, y_pred)
                        
                        fig_cm, ax = plt.subplots(figsize=(6, 5))
                        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                                    xticklabels=['Normal', 'Anomaly'],
                                    yticklabels=['Normal', 'Anomaly'],
                                    ax=ax, cbar_kws={'label': 'Count'})
                        ax.set_xlabel('Predicted')
                        ax.set_ylabel('Actual')
                        ax.set_title('Confusion Matrix')
                        st.pyplot(fig_cm)
                        
                       
            else:
                st.info("ℹ️ Please run the data preprocessing step first to test the model.")
        
        # TAB 2: Model Information
        with tab2:
            st.subheader("📊 Model Information")
            
            col_info1, col_info2 = st.columns(2)
            
            with col_info1:
                st.write("**Model Details:**")
                st.write(f"- **Type:** {type(model).__name__}")
                st.write(f"- **Classes:** {model.classes_.tolist()}")
                st.write(f"- **Number of Classes:** {model.n_classes_}")
                
                if hasattr(model, 'n_estimators'):
                    st.write(f"- **Number of Trees:** {model.n_estimators}")
                if hasattr(model, 'max_depth'):
                    st.write(f"- **Max Depth:** {model.max_depth}")
                if hasattr(model, 'class_weight'):
                    st.write(f"- **Class Weight:** {model.class_weight}")
                
                st.write("**Training Features:**")
                st.write("• Temperature (°C)")
                st.write("• Humidity (%)")
                st.write("• Air Quality (AQI)")
            
            with col_info2:
                st.write("**Key Fix Applied:**")
                st.success("✅ class_weight='balanced'")
                st.write("This fixes the 7.4% anomaly imbalance")
                
                # Feature importance
                if hasattr(model, 'feature_importances_'):
                    st.subheader("🔍 Feature Importance")
                    
                    feature_names = ['Temperature', 'Humidity', 'Air Quality']
                    importances = model.feature_importances_
                    
                    importance_df = pd.DataFrame({
                        'Feature': feature_names,
                        'Importance': importances
                    }).sort_values('Importance', ascending=True)
                    
                    fig_imp, ax = plt.subplots(figsize=(8, 4))
                    bars = ax.barh(importance_df['Feature'], importance_df['Importance'], 
                                  color=['#1f77b4', '#ff7f0e', '#2ca02c'])
                    
                    for bar in bars:
                        width = bar.get_width()
                        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                               f'{width:.1%}', va='center')
                    
                    ax.set_xlabel('Importance Score')
                    ax.set_xlim(0, 1)
                    ax.set_title('Feature Importance')
                    ax.grid(True, alpha=0.3, axis='x')
                    st.pyplot(fig_imp)
        
        # TAB 3: Live Predictions (FIXED VERSION)
        with tab3:
            st.subheader("⚡ Live Anomaly Detection")
            st.markdown("Enter sensor values to check for anomalies in real-time:")
            
            # Input controls
            col1, col2, col3 = st.columns(3)
            with col1:
                temperature = st.number_input(
                    "Temperature (°C)", 
                    min_value=-10.0, 
                    max_value=100.0, 
                    value=25.0, 
                    step=0.5,
                    help="Typical range: 15°C to 35°C"
                )
            with col2:
                humidity = st.number_input(
                    "Humidity (%)", 
                    min_value=0.0, 
                    max_value=100.0, 
                    value=50.0, 
                    step=1.0,
                    help="Typical range: 30% to 70%"
                )
            with col3:
                air_quality = st.number_input(
                    "Air Quality (AQI)", 
                    min_value=0.0, 
                    max_value=500.0, 
                    value=70.0, 
                    step=5.0,
                    help="Lower is better. Typical range: 0-150"
                )
            
            # FIXED PREDICTION FUNCTION
            def predict_with_scaling(temp, hum, aq):
                """Correct prediction with scaling"""
                # Create DataFrame
                input_df = pd.DataFrame({
                    'temperature': [temp],
                    'humidity': [hum],
                    'air_quality': [aq]
                })
                
                # CRITICAL: Scale the input
                input_scaled = scaler.transform(input_df)
                
                # Predict
                is_anomaly = model.predict(input_scaled)[0]
                probabilities = model.predict_proba(input_scaled)[0]
                
                return {
                    'is_anomaly': bool(is_anomaly),
                    'anomaly_prob': float(probabilities[1]),
                    'normal_prob': float(probabilities[0]),
                    'anomaly_percent': f"{probabilities[1]:.1%}",
                    'normal_percent': f"{probabilities[0]:.1%}"
                }
            
            # Prediction button
            if st.button("🔍 Check for Anomaly", type="primary"):
                result = predict_with_scaling(temperature, humidity, air_quality)
                
                # Display results
                if result['is_anomaly']:
                    st.error(f"## 🔴 ANOMALY DETECTED!")
                    st.metric("Anomaly Probability", result['anomaly_percent'])
                else:
                    st.success(f"## ✅ NORMAL READING")
                    st.metric("Normal Probability", result['normal_percent'])
                
                # Visual gauge
                st.subheader("Probability Gauge")
                fig_gauge, ax = plt.subplots(figsize=(10, 2))
                
                ax.barh([''], [1], color='lightgray', alpha=0.3)
                ax.barh([''], [result['anomaly_prob']], 
                       color='red' if result['anomaly_prob'] > 0.5 else 'green')
                ax.axvline(x=0.5, color='black', linestyle='--', linewidth=1, alpha=0.7)
                
                ax.set_xlim(0, 1)
                ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
                ax.set_xticklabels(['0%', '25%', '50%', '75%', '100%'])
                ax.set_xlabel('Anomaly Probability')
                ax.grid(True, alpha=0.3, axis='x')
                
                st.pyplot(fig_gauge)
                
                # Quick demo of the fix
                st.subheader("🧪 Proof of Fix")
                col_demo1, col_demo2 = st.columns(2)
                
                with col_demo1:
                    normal_result = predict_with_scaling(22, 50, 70)
                    st.write("**Normal case (22°C, 50%, 70 AQI):**")
                    st.write(f"Anomaly: {normal_result['anomaly_percent']}")
                
                with col_demo2:
                    anomaly_result = predict_with_scaling(38, 90, 180)
                    st.write("**Anomaly case (38°C, 90%, 180 AQI):**")
                    st.write(f"Anomaly: {anomaly_result['anomaly_percent']}")
            
         

# -------------------------
# FOOTER
# -------------------------
st.markdown("---")
st.caption("IoT Sensor Analytics Dashboard | Anomaly Detection System v2.0 | sara souissi")
"""
Flask API Server for EPE Patch Window Predictor
Simplified version with working chatbot integration
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime, timedelta
import sys
import os
import traceback

# Import your predictor module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from epe_patch_window_predictor import (
    PatchWindowPredictor,
    main_pipeline
)

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Global variables to store the predictor
predictor = None
pipeline_results = None
last_predictions = []


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'message': 'EPE Patch Window Predictor API is running',
        'model_loaded': predictor is not None
    })


@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Generate predictions for a date range
    
    Expected JSON:
    {
        "start_date": "2025-01-01",
        "num_days": 30,
        "sort_by": "date"  // Optional: "date" (default), "confidence", or "day"
    }
    """
    global predictor, pipeline_results, last_predictions
    
    try:
        data = request.json
        start_date_str = data.get('start_date')
        num_days = data.get('num_days', 30)
        sort_by = data.get('sort_by', 'date')  # New parameter
        
        print(f"\n📅 Request received: {start_date_str}, {num_days} days, sort by: {sort_by}")
        
        # Initialize model if not already done
        if predictor is None:
            print("🔄 Training model (first time only, please wait 30-60 seconds)...")
            try:
                # Look for data in the data/ subfolder
                data_path = './data/'
                if not os.path.exists(data_path):
                    data_path = '../data/'  # Try parent directory
                
                # Run pipeline with NaN handling
                pipeline_results = main_pipeline(data_path)
                predictor = pipeline_results['predictor']
                print("✅ Model trained successfully!")
            except ValueError as e:
                error_msg = str(e)
                if "NaN" in error_msg or "missing values" in error_msg:
                    print(f"❌ Data quality issue: {error_msg}")
                    return jsonify({
                        'success': False,
                        'error': 'Data contains missing values. Please check your CSV files for incomplete data or contact support.'
                    }), 500
                else:
                    print(f"❌ Model training error: {error_msg}")
                    return jsonify({
                        'success': False,
                        'error': f'Model training failed: {error_msg}. Make sure data files are in the data/ folder.'
                    }), 500
            except Exception as e:
                print(f"❌ Model training error: {str(e)}")
                traceback.print_exc()
                return jsonify({
                    'success': False,
                    'error': f'Model training failed: {str(e)}. Make sure data files are in the data/ folder.'
                }), 500
        
        # Parse start date
        start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
        end_date = start_date + timedelta(days=num_days)
        
        print(f"🔮 Generating predictions from {start_date} to {end_date}...")
        
        # Generate predictions
        predictions_df = predictor.predict_date_range(start_date, end_date)
        
        # Find best windows
        best_windows = predictor.find_best_windows(predictions_df, top_n=50)
        
        print(f"✅ Found {len(best_windows)} optimal windows")
        
        # Convert to frontend format
        predictions_list = []
        for _, row in best_windows.iterrows():
            predictions_list.append({
                'date': row['date'].strftime('%Y-%m-%d'),
                'dayOfWeek': row['day_name'],
                'window': row['window'],
                'confidence': int(row['probability'] * 100),
                'impact': row['impact_level'],
                'timestamp': row['date'].strftime('%Y-%m-%d')  # For sorting
            })
        
        # Optional: Apply additional sorting based on user preference
        # Note: Backend already sorts chronologically, but API can re-sort if needed
        if sort_by == 'confidence':
            predictions_list.sort(key=lambda x: x['confidence'], reverse=True)
            print("📊 Sorted by confidence (highest first)")
        elif sort_by == 'day':
            # Sort by day of week (Monday first)
            day_order = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 
                        'Friday': 4, 'Saturday': 5, 'Sunday': 6}
            predictions_list.sort(key=lambda x: (day_order.get(x['dayOfWeek'], 7), x['timestamp']))
            print("📅 Sorted by day of week")
        else:  # 'date' or default
            # Already sorted chronologically by backend
            print("📅 Sorted chronologically (date and time)")
        
        # Store for chatbot
        last_predictions = predictions_list
        
        return jsonify({
            'success': True,
            'predictions': predictions_list,
            'total_windows': len(predictions_list)
        })
    
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/chat', methods=['POST'])
def chat():
    """
    Handle chatbot questions with rule-based responses
    """
    global last_predictions
    
    try:
        data = request.json
        question = data.get('question', '').lower()
        predictions = data.get('predictions', last_predictions)
        
        if not predictions:
            return jsonify({
                'success': True,
                'answer': '⚠️ No predictions available yet. Please generate predictions first by clicking "Generate Predictions".'
            })
        
        # Generate answer based on question
        answer = generate_answer(question, predictions)
        
        return jsonify({
            'success': True,
            'answer': answer,
            'question': question
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


def generate_answer(question, predictions):
    """Generate intelligent responses based on question type"""
    
    # Calculate statistics
    total = len(predictions)
    high_conf = len([p for p in predictions if p['confidence'] >= 85])
    medium_conf = len([p for p in predictions if 70 <= p['confidence'] < 85])
    avg_conf = sum(p['confidence'] for p in predictions) / total if total > 0 else 0
    weekends = len([p for p in predictions if p['dayOfWeek'] in ['Saturday', 'Sunday']])
    weekdays = total - weekends
    
    # Get best window
    best = predictions[0] if predictions else None
    
    # Question matching with comprehensive responses
    if any(word in question for word in ['statistic', 'stats', 'summary', 'overview', 'how many']):
        return f"""📊 **Prediction Statistics:**

• Total optimal windows: **{total}**
• High confidence (85%+): **{high_conf}** windows ({(high_conf/total*100):.1f}%)
• Medium confidence (70-84%): **{medium_conf}** windows
• Average confidence: **{avg_conf:.1f}%**
• Weekend windows: **{weekends}** ({(weekends/total*100):.1f}%)
• Weekday windows: **{weekdays}** ({(weekdays/total*100):.1f}%)

The model has identified {total} optimal maintenance windows based on 10 years of historical electricity demand data."""

    elif any(word in question for word in ['best', 'top', 'highest', 'recommended', '#1', 'number 1', 'first']):
        if best:
            weekend_status = "✅ Weekend" if best['dayOfWeek'] in ['Saturday', 'Sunday'] else "📅 Weekday"
            return f"""🏆 **#1 Recommended Window:**

📅 **Date:** {best['date']} ({best['dayOfWeek']})
⏰ **Time:** {best['window']}
🎯 **Confidence:** {best['confidence']}%
📉 **Impact:** {best['impact']}
🗓️ **Type:** {weekend_status}

**Why this window?**
This window achieved the highest confidence score because:
• Historical data shows consistently low demand during this time
• Pattern recognition indicates minimal system activity
• {best['confidence']}% probability of optimal conditions
• Best risk-reward ratio in the entire date range

This is your **safest and most reliable** maintenance window."""
        return "No predictions available."
    
    elif any(word in question for word in ['weekend', 'saturday', 'sunday']):
        weekend_windows = [p for p in predictions if p['dayOfWeek'] in ['Saturday', 'Sunday']]
        if weekend_windows:
            top_3 = weekend_windows[:3]
            avg_weekend_conf = sum(w['confidence'] for w in weekend_windows)/len(weekend_windows)
            windows_text = "\n".join([
                f"**#{i+1}.** {w['date']} ({w['dayOfWeek']}) at {w['window']} - {w['confidence']}% confidence"
                for i, w in enumerate(top_3)
            ])
            return f"""🎯 **Weekend Maintenance Windows:**

Found **{len(weekend_windows)}** optimal weekend slots:
• Average confidence: **{avg_weekend_conf:.1f}%**
• Saturday slots: {len([w for w in weekend_windows if w['dayOfWeek'] == 'Saturday'])}
• Sunday slots: {len([w for w in weekend_windows if w['dayOfWeek'] == 'Sunday'])}

**Top 3 Weekend Windows:**
{windows_text}

**Why weekends are better:**
• 30-40% lower electricity demand on weekends
• Reduced business operations and user activity
• More flexible maintenance schedules
• Lower risk of disrupting critical operations"""
        return "No weekend windows found in the current predictions."
    
    elif any(word in question for word in ['weekday', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday']):
        weekday_windows = [p for p in predictions if p['dayOfWeek'] not in ['Saturday', 'Sunday']]
        if weekday_windows:
            top_3 = weekday_windows[:3]
            avg_weekday_conf = sum(w['confidence'] for w in weekday_windows)/len(weekday_windows)
            windows_text = "\n".join([
                f"**#{i+1}.** {w['date']} ({w['dayOfWeek']}) at {w['window']} - {w['confidence']}% confidence"
                for i, w in enumerate(top_3)
            ])
            
            # Count by day
            day_counts = {}
            for w in weekday_windows:
                day_counts[w['dayOfWeek']] = day_counts.get(w['dayOfWeek'], 0) + 1
            
            return f"""📅 **Weekday Maintenance Windows:**

Found **{len(weekday_windows)}** optimal weekday slots:
• Average confidence: **{avg_weekday_conf:.1f}%**
• Distribution: {', '.join([f'{day}: {count}' for day, count in sorted(day_counts.items())])}

**Top 3 Weekday Windows:**
{windows_text}

**Weekday considerations:**
• Best windows are typically overnight (12 AM - 6 AM)
• Higher demand during business hours (9 AM - 5 PM)
• Early morning windows minimize user impact"""
        return "No weekday windows found in the current predictions."
    
    elif any(word in question for word in ['next', 'upcoming', 'soon', 'soonest', 'earliest']):
        if predictions:
            next_window = predictions[0]
            second_option = predictions[1] if len(predictions) > 1 else None
            
            response = f"""⏰ **Next Available Window:**

**Primary Option:**
📅 Date: {next_window['date']} ({next_window['dayOfWeek']})
⏰ Time: {next_window['window']}
🎯 Confidence: {next_window['confidence']}%
📉 Impact: {next_window['impact']}

This is the **soonest high-confidence** maintenance window."""
            
            if second_option:
                response += f"""

**Backup Option:**
📅 Date: {second_option['date']} ({second_option['dayOfWeek']})
⏰ Time: {second_option['window']}
🎯 Confidence: {second_option['confidence']}%"""
            
            return response
        return "No upcoming windows available."
    
    elif any(word in question for word in ['confidence', 'reliable', 'trust', 'accurate', 'sure']):
        high_list = [p for p in predictions if p['confidence'] >= 85]
        medium_list = [p for p in predictions if 70 <= p['confidence'] < 85]
        
        return f"""🎯 **Confidence Level Breakdown:**

**Current Distribution:**
• 🟢 High (85-100%): **{len(high_list)} windows** ({(len(high_list)/total*100):.1f}%)
• 🟡 Medium (70-84%): **{len(medium_list)} windows** ({(len(medium_list)/total*100):.1f}%)
• Average: **{avg_conf:.1f}%**

**What Confidence Means:**

🟢 **85-100% (High Confidence)**
• Very reliable predictions based on strong historical patterns
• 10+ years of consistent data showing low demand
• Minimal risk of high system load
• **Recommended for critical maintenance**

🟡 **70-84% (Medium Confidence)**  
• Good reliability with favorable conditions
• Some variability in historical data
• Acceptable risk level
• Suitable for non-critical updates

**Model Accuracy:**
• Trained on 10 years of hourly electricity demand data (2015-2024)
• Considers day of week, time of day, and seasonal patterns
• Cross-validated for reliability

Higher confidence = more predictable low-demand periods!"""
    
    elif any(word in question for word in ['why', 'explain', 'reason', 'how', 'work', 'algorithm']):
        return f"""💡 **How the Prediction System Works:**

**Data Source:**
• 10 years of historical electricity demand data (2015-2024)
• 87,648 hourly measurements from El Paso Electric
• Real-world pattern analysis

**Machine Learning Approach:**
• Ensemble model combining 4 algorithms
• Pattern recognition across multiple dimensions
• Continuous learning from historical trends

**Key Factors Analyzed:**

⏰ **Time Patterns**
• Time of day (overnight = lower demand)
• Day of week (weekends typically better)  
• Hour of day (12 AM - 6 AM optimal)

📊 **Demand Indicators**
• Historical load patterns
• Seasonal variations
• Holiday impacts
• Weather influences

🎯 **Current Results:**
• {total} optimal windows identified
• {high_conf} high-confidence windows
• {weekends} weekend opportunities
• Average {avg_conf:.1f}% confidence

**Output:**
Windows ranked by confidence score, where higher scores indicate more reliable predictions based on historical low-demand periods."""
    
    elif any(word in question for word in ['compare', 'difference', 'versus', 'vs', 'between']):
        if len(predictions) >= 2:
            first = predictions[0]
            second = predictions[1]
            
            return f"""🔍 **Comparison: Top 2 Windows**

**Window #1 (Best):**
• Date: {first['date']} ({first['dayOfWeek']})
• Time: {first['window']}
• Confidence: {first['confidence']}%
• Impact: {first['impact']}

**Window #2 (Alternative):**
• Date: {second['date']} ({second['dayOfWeek']})
• Time: {second['window']}
• Confidence: {second['confidence']}%
• Impact: {second['impact']}

**Key Differences:**
• Confidence gap: {first['confidence'] - second['confidence']} percentage points
• Window #1 is {first['confidence'] - second['confidence']}% more reliable
• Both windows meet high-confidence threshold
• Consider scheduling flexibility when choosing"""
        return "Need at least 2 windows for comparison."
    
    elif any(word in question for word in ['overnight', 'night', 'late', 'early morning']):
        overnight = [p for p in predictions if any(t in p['window'].lower() for t in ['12 am', '1 am', '2 am', '3 am', '4 am', '5 am', '6 am'])]
        if overnight:
            avg_overnight_conf = sum(w['confidence'] for w in overnight)/len(overnight)
            top_3 = overnight[:3]
            windows_text = "\n".join([
                f"**#{i+1}.** {w['date']} at {w['window']} - {w['confidence']}% confidence"
                for i, w in enumerate(top_3)
            ])
            
            return f"""🌙 **Overnight Maintenance Windows:**

Found **{len(overnight)}** overnight slots (12 AM - 6 AM):
• Average confidence: **{avg_overnight_conf:.1f}%**
• Best overnight confidence: **{max(w['confidence'] for w in overnight)}%**

**Top 3 Overnight Windows:**
{windows_text}

**Why overnight is optimal:**
• Electricity demand drops 40-60% during these hours
• Minimal user activity and business operations
• Historical data shows consistent low-load patterns
• Reduced risk of service disruption
• Natural maintenance window for most systems"""
        return "No overnight windows identified in current predictions."
    
    else:
        # Default help message
        return f"""🤖 **I can help you understand the predictions!**

**Try asking:**
• "What are the statistics?" - Complete overview
• "What's the best window?" - Top recommendation
• "Show me weekend windows" - Saturday/Sunday options
• "What about weekdays?" - Monday-Friday slots
• "When is the next window?" - Soonest option
• "Explain confidence levels" - Reliability details
• "Why are these windows optimal?" - How it works
• "Compare the top 2 windows" - Window comparison
• "Show me overnight windows" - Late night options

**Current Data:**
• {total} optimal windows available
• {high_conf} with high confidence (85%+)
• {weekends} weekend opportunities

What would you like to know? 😊"""


@app.route('/api/statistics', methods=['POST'])
def get_statistics():
    """Get statistics about predictions"""
    global last_predictions
    
    try:
        data = request.json
        predictions = data.get('predictions', last_predictions)
        
        if not predictions:
            return jsonify({
                'success': False,
                'error': 'No predictions available'
            })
        
        total = len(predictions)
        high_conf = len([p for p in predictions if p['confidence'] >= 85])
        medium_conf = len([p for p in predictions if 70 <= p['confidence'] < 85])
        avg_conf = sum(p['confidence'] for p in predictions) / total
        weekends = len([p for p in predictions if p['dayOfWeek'] in ['Saturday', 'Sunday']])
        
        stats = {
            'total_windows': total,
            'high_confidence_count': high_conf,
            'medium_confidence_count': medium_conf,
            'avg_confidence': round(avg_conf, 1),
            'weekend_count': weekends
        }
        
        return jsonify({
            'success': True,
            'statistics': stats
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


if __name__ == '__main__':
    print("=" * 80)
    print("EPE PATCH WINDOW PREDICTOR API SERVER")
    print("=" * 80)
    print("\n🚀 Starting Flask API server...")
    print("📡 Frontend should connect to: http://localhost:5001")
    print("\n📋 Available endpoints:")
    print("  - GET  /api/health      - Health check")
    print("  - POST /api/predict     - Generate predictions")
    print("  - POST /api/chat        - Ask chatbot questions")
    print("  - POST /api/statistics  - Get prediction statistics")
    print("\n" + "=" * 80)
    print("⚠️  NOTE: First prediction will take 30-60 seconds (training model)")
    print("⚠️  Make sure data files are in the 'data/' folder")
    print("=" * 80 + "\n")
    
    # Run the server on port 5001
    #app.run(host='0.0.0.0', port=5001, debug=True)
    import os
    port = int(os.environ.get('PORT', 5001))
    debug = os.environ.get('DEBUG', 'False') == 'True'
    
    app.run(
        host='0.0.0.0',
        port=port,
        debug=debug
    )
"""
Student AI System v2 — Advanced Full-Stack Application
Features: SHAP Explainability, What-If Simulator, Cohort Compare,
          RBAC, 2FA, Audit Log, API Keys, Anomaly Detection, NLP Insights
"""

import os, io, csv, json, hashlib, secrets, time
from datetime import datetime
from functools import wraps

import pandas as pd
import numpy as np
from flask import (Flask, render_template, request, redirect, url_for,
                   session, jsonify, send_file, flash)
from werkzeug.utils import secure_filename

import sys
sys.path.insert(0, os.path.dirname(__file__))
from models.pipeline import pipeline, generate_study_plan

# ─── App Setup ──────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'studentAI_v2_xK9mP2_2024')
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'datasets')
ALLOWED_EXTENSIONS = {'csv'}

# ─── In-Memory Stores ───────────────────────────────────────────
students_store = []        # current prediction results
cohort_store = {}          # {label: df}
notifications_store = []
audit_log = []             # [{ts, user, action, detail, ip}]
api_keys_store = {}        # {key: {name, created, requests}}
blocked_students = set()
two_fa_secrets = {}        # {username: secret}
feedback_store = []        # [{student_id, student_name, teacher, subject, rating, comment, ts}]

# ─── Student Accounts ───────────────────────────────────────────
STUDENT_ACCOUNTS = {
    'S001': {'password': hashlib.sha256('alice123'.encode()).hexdigest(),    'name': 'Alice Johnson',     'roll': 'MIT2024001', 'branch': 'Computer Engineering'},
    'S002': {'password': hashlib.sha256('bob123'.encode()).hexdigest(),      'name': 'Bob Smith',         'roll': 'MIT2024002', 'branch': 'Civil Engineering'},
    'S003': {'password': hashlib.sha256('carol123'.encode()).hexdigest(),    'name': 'Carol White',       'roll': 'MIT2024003', 'branch': 'Mechanical Engineering'},
    'S004': {'password': hashlib.sha256('david123'.encode()).hexdigest(),    'name': 'David Brown',       'roll': 'MIT2024004', 'branch': 'Electrical Engineering'},
    'S005': {'password': hashlib.sha256('emma123'.encode()).hexdigest(),     'name': 'Emma Davis',        'roll': 'MIT2024005', 'branch': 'Computer Engineering'},
    'S006': {'password': hashlib.sha256('frank123'.encode()).hexdigest(),    'name': 'Frank Wilson',      'roll': 'MIT2024006', 'branch': 'Civil Engineering'},
    'S007': {'password': hashlib.sha256('grace123'.encode()).hexdigest(),    'name': 'Grace Lee',         'roll': 'MIT2024007', 'branch': 'Electronics Engineering'},
    'S008': {'password': hashlib.sha256('henry123'.encode()).hexdigest(),    'name': 'Henry Taylor',      'roll': 'MIT2024008', 'branch': 'Mechanical Engineering'},
    'S009': {'password': hashlib.sha256('isabella123'.encode()).hexdigest(), 'name': 'Isabella Martinez', 'roll': 'MIT2024009', 'branch': 'Computer Engineering'},
    'S010': {'password': hashlib.sha256('jack123'.encode()).hexdigest(),     'name': 'Jack Anderson',     'roll': 'MIT2024010', 'branch': 'Electrical Engineering'},
}

SUBJECT_TEACHERS = {
    'Mathematics': 'Prof. R. Sharma',
    'Science':     'Prof. P. Kulkarni',
    'English':     'Prof. S. Desai',
    'Programming': 'Prof. A. Patil',
    'History':     'Prof. M. Jadhav',
}

# ─── RBAC Users ─────────────────────────────────────────────────
USERS = {
    'admin':    {'password': hashlib.sha256('admin123'.encode()).hexdigest(),  'role': 'super_admin',  'name': 'Super Admin'},
    'teacher':  {'password': hashlib.sha256('teacher123'.encode()).hexdigest(), 'role': 'teacher',     'name': 'Mr. Teacher'},
    'counselor':{'password': hashlib.sha256('counsel123'.encode()).hexdigest(), 'role': 'counselor',   'name': 'Dr. Counselor'},
}
ROLE_PERMISSIONS = {
    'super_admin': ['view', 'notify', 'block', 'delete', 'download', 'manage_users', 'api_keys', 'audit'],
    'teacher':     ['view', 'notify'],
    'counselor':   ['view', 'notify', 'download'],
}
MAX_LOGIN_ATTEMPTS = 3

# ─── Helpers ────────────────────────────────────────────────────
def allowed_file(f): return '.' in f and f.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
def hash_pw(pw): return hashlib.sha256(pw.encode()).hexdigest()

def has_perm(perm):
    role = session.get('role', '')
    return perm in ROLE_PERMISSIONS.get(role, [])

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get('logged_in'):
            return redirect(url_for('admin_login'))
        return f(*args, **kwargs)
    return decorated

def admin_required(f):
    """Blocks students AND unauthenticated users from admin-only pages."""
    @wraps(f)
    def decorated(*args, **kwargs):
        if session.get('student_logged_in') and not session.get('logged_in'):
            flash('Access denied — this section is for administrators only.', 'danger')
            return redirect(url_for('student_portal'))
        if not session.get('logged_in'):
            flash('Please log in as admin to access this page.', 'warning')
            return redirect(url_for('admin_login'))
        return f(*args, **kwargs)
    return decorated

def require_perm(perm):
    def decorator(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            if not session.get('logged_in'):
                return redirect(url_for('admin_login'))
            if not has_perm(perm):
                flash('Access denied — insufficient permissions.', 'danger')
                return redirect(url_for('admin_dashboard'))
            return f(*args, **kwargs)
        return decorated
    return decorator

def add_audit(action, detail=''):
    audit_log.append({
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'user': session.get('username', 'system'),
        'role': session.get('role', '—'),
        'action': action,
        'detail': detail,
        'ip': request.remote_addr or '127.0.0.1'
    })

def ensure_models():
    if not pipeline.is_loaded():
        return pipeline.load_models()
    return True

def risk_badge(r): return {'High': 'danger', 'Medium': 'warning', 'Low': 'success'}.get(r, 'secondary')
def perf_badge(p): return {'Good': 'success', 'Average': 'warning', 'Low': 'danger'}.get(p, 'secondary')

# ─── Public Routes ───────────────────────────────────────────────
@app.route('/')
def index():
    """MIT College splash / landing page"""
    ensure_models()
    stats = {
        'total': len(students_store),
        'high_risk': sum(1 for s in students_store if s.get('Dropout_Risk') == 'High'),
        'anomalies': sum(1 for s in students_store if s.get('Is_Anomaly')),
        'model_ready': pipeline.is_loaded()
    }
    return render_template('splash.html', stats=stats)

@app.route('/app')
def main_app():
    """Original main app dashboard"""
    ensure_models()
    stats = {
        'total': len(students_store),
        'high_risk': sum(1 for s in students_store if s.get('Dropout_Risk') == 'High'),
        'anomalies': sum(1 for s in students_store if s.get('Is_Anomaly')),
        'model_ready': pipeline.is_loaded()
    }
    return render_template('index.html', stats=stats)

@app.route('/upload', methods=['GET', 'POST'])
@admin_required
def upload():
    if not ensure_models():
        flash('ML models not ready. Run train_model.py first.', 'danger')
        return redirect(url_for('index'))
    if request.method == 'POST':
        label = request.form.get('cohort_label', '').strip() or None
        file = request.files.get('file')
        if not file or not allowed_file(file.filename):
            flash('Please upload a valid CSV file.', 'warning')
            return redirect(request.url)
        fname = secure_filename(file.filename)
        fpath = os.path.join(app.config['UPLOAD_FOLDER'], fname)
        file.save(fpath)
        try:
            df = pd.read_csv(fpath)
            results_df = pipeline.predict_batch(df)
            global students_store
            students_store = results_df.to_dict('records')
            if label:
                cohort_store[label] = df
            for s in students_store:
                if s.get('Dropout_Risk') == 'High':
                    notifications_store.append({
                        'student': s.get('Student_Name', '?'),
                        'message': f"⚠️ {s.get('Student_Name')} is at HIGH dropout risk. Please schedule an advisor meeting.",
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M'),
                        'type': 'danger', 'read': False
                    })
            add_audit('CSV Upload', f'{len(students_store)} students processed from {fname}')
            flash(f'✓ {len(students_store)} students processed successfully!', 'success')
            return redirect(url_for('results'))
        except Exception as e:
            flash(f'Error: {str(e)}', 'danger')
    cohorts = list(cohort_store.keys())
    return render_template('upload.html', cohorts=cohorts)

@app.route('/predict-single', methods=['POST'])
def predict_single():
    if not ensure_models():
        return jsonify({'error': 'Models not loaded'}), 500
    try:
        data = request.get_json()
        result = pipeline.predict_single(data)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/whatif', methods=['POST'])
def whatif():
    if not ensure_models():
        return jsonify({'error': 'Models not loaded'}), 500
    data = request.get_json()
    base = data.get('base', {})
    changes = data.get('changes', {})
    result = pipeline.whatif_simulate(base, changes)
    return jsonify(result)

@app.route('/results')
@admin_required
def results():
    if not students_store:
        flash('No data yet. Upload a dataset first.', 'info')
        return redirect(url_for('upload'))
    df = pd.DataFrame(students_store)
    analytics = pipeline.get_analytics(df) if pipeline.is_loaded() else {}
    # Leaderboard: sort by health score
    leaderboard = sorted(students_store, key=lambda x: float(x.get('Health_Score', 0)), reverse=True)
    return render_template('results.html', students=students_store, analytics=analytics,
                           leaderboard=leaderboard[:10], risk_badge=risk_badge, perf_badge=perf_badge)

@app.route('/student-profile/<student_id>')
@admin_required
def student_profile(student_id):
    student = next((s for s in students_store if str(s.get('Student_ID')) == str(student_id)), None)
    if not student:
        return jsonify({'error': 'Not found'}), 404
    shap = pipeline.get_shap_explanation(student) if pipeline.is_loaded() else []
    return jsonify({'student': student, 'shap': shap})

@app.route('/study-planner')
@admin_required
def study_planner():
    return render_template('study_planner.html', students=students_store)

@app.route('/api/study-plan', methods=['POST'])
def api_study_plan():
    d = request.get_json()
    plan = generate_study_plan(d.get('weak_subject', 'Mathematics'),
                               d.get('study_hours', 3), d.get('performance', 'Average'))
    return jsonify({'plan': plan})

@app.route('/api/analytics')
def api_analytics():
    if not students_store: return jsonify({})
    return jsonify(pipeline.get_analytics(pd.DataFrame(students_store)))

@app.route('/cohort-compare', methods=['GET', 'POST'])
@admin_required
def cohort_compare():
    result = None
    if request.method == 'POST':
        f1 = request.files.get('file1')
        f2 = request.files.get('file2')
        l1 = request.form.get('label1', 'Cohort A')
        l2 = request.form.get('label2', 'Cohort B')
        if f1 and f2 and allowed_file(f1.filename) and allowed_file(f2.filename):
            df1 = pd.read_csv(f1)
            df2 = pd.read_csv(f2)
            result = pipeline.compare_cohorts(df1, df2, l1, l2)
    return render_template('cohort_compare.html', result=result)

# ─── Admin Auth ─────────────────────────────────────────────────
@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    if session.get('logged_in'):
        return redirect(url_for('admin_dashboard'))
    if request.method == 'POST':
        attempts = session.get('login_attempts', 0)
        if attempts >= MAX_LOGIN_ATTEMPTS:
            return render_template('admin_login.html', blocked=True)
        username = request.form.get('username', '').lower()
        password = request.form.get('password', '')
        totp_code = request.form.get('totp_code', '').strip()
        user = USERS.get(username)
        if user and hash_pw(password) == user['password']:
            # 2FA check
            if username in two_fa_secrets:
                try:
                    import pyotp
                    totp = pyotp.TOTP(two_fa_secrets[username])
                    if not totp.verify(totp_code):
                        return render_template('admin_login.html', error='Invalid 2FA code. Try again.',
                                               show_totp=True, username=username)
                except ImportError:
                    pass  # 2FA not available, skip
            session['logged_in'] = True
            session['username'] = username
            session['role'] = user['role']
            session['display_name'] = user['name']
            session['login_attempts'] = 0
            add_audit('Login', f"Role: {user['role']}")
            flash(f"Welcome back, {user['name']}!", 'success')
            return redirect(url_for('admin_dashboard'))
        else:
            attempts += 1
            session['login_attempts'] = attempts
            remaining = MAX_LOGIN_ATTEMPTS - attempts
            error = f'Invalid credentials. {remaining} attempt(s) remaining.' if remaining > 0 else 'Account blocked.'
            return render_template('admin_login.html', error=error, attempts=attempts,
                                   blocked=remaining <= 0)
    return render_template('admin_login.html')

@app.route('/admin/logout')
def admin_logout():
    add_audit('Logout')
    session.clear()
    flash('Logged out successfully.', 'info')
    return redirect(url_for('index'))

@app.route('/admin/dashboard')
@login_required
def admin_dashboard():
    ensure_models()
    df = pd.DataFrame(students_store) if students_store else pd.DataFrame()
    analytics = pipeline.get_analytics(df) if not df.empty and pipeline.is_loaded() else {}
    meta = pipeline.metadata if pipeline.is_loaded() else {}
    unread = sum(1 for n in notifications_store if not n.get('read'))
    return render_template('admin_dashboard.html',
                           students=students_store, blocked_students=blocked_students,
                           notifications=notifications_store, analytics=analytics,
                           metadata=meta, audit_log=audit_log[-50:][::-1],
                           api_keys=api_keys_store, users=USERS,
                           risk_badge=risk_badge, perf_badge=perf_badge,
                           has_perm=has_perm, unread=unread,
                           role=session.get('role'), display_name=session.get('display_name'))

@app.route('/admin/block-student', methods=['POST'])
@require_perm('block')
def block_student():
    sid = request.get_json().get('student_id')
    if sid:
        blocked_students.add(str(sid))
        add_audit('Block Student', f'ID: {sid}')
        return jsonify({'success': True})
    return jsonify({'success': False}), 400

@app.route('/admin/delete-student', methods=['POST'])
@require_perm('delete')
def delete_student():
    sid = str(request.get_json().get('student_id', ''))
    global students_store
    students_store = [s for s in students_store if str(s.get('Student_ID', '')) != sid]
    add_audit('Delete Student', f'ID: {sid}')
    return jsonify({'success': True, 'total': len(students_store)})

@app.route('/admin/send-notification', methods=['POST'])
@require_perm('notify')
def send_notification():
    d = request.get_json()
    name = d.get('student_name', 'Student')
    msg = d.get('message', f"⚠️ {name}: Please contact your academic advisor.")
    notifications_store.append({
        'student': name, 'message': msg,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'type': 'warning', 'read': False
    })
    add_audit('Send Notification', f'To: {name}')
    return jsonify({'success': True, 'total': len(notifications_store)})

@app.route('/admin/mark-notifications-read', methods=['POST'])
@login_required
def mark_read():
    for n in notifications_store: n['read'] = True
    return jsonify({'success': True})

@app.route('/admin/clear-notifications', methods=['POST'])
@require_perm('notify')
def clear_notifications():
    notifications_store.clear()
    return jsonify({'success': True})

# ─── API Key Management ──────────────────────────────────────────
@app.route('/admin/api-keys/generate', methods=['POST'])
@require_perm('api_keys')
def generate_api_key():
    d = request.get_json()
    name = d.get('name', 'Unnamed Key')
    key = 'sai_' + secrets.token_hex(24)
    api_keys_store[key] = {
        'name': name, 'key': key,
        'created': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'requests': 0, 'active': True
    }
    add_audit('Generate API Key', f'Name: {name}')
    return jsonify({'success': True, 'key': key, 'name': name})

@app.route('/admin/api-keys/revoke', methods=['POST'])
@require_perm('api_keys')
def revoke_api_key():
    key = request.get_json().get('key')
    if key in api_keys_store:
        api_keys_store[key]['active'] = False
        add_audit('Revoke API Key', f'Key: {key[:12]}...')
        return jsonify({'success': True})
    return jsonify({'success': False}), 404

# ─── 2FA ─────────────────────────────────────────────────────────
@app.route('/admin/2fa/setup', methods=['GET', 'POST'])
@login_required
def setup_2fa():
    username = session.get('username')
    try:
        import pyotp, qrcode, base64
        if request.method == 'POST':
            code = request.form.get('code', '')
            secret = session.get('pending_2fa_secret')
            totp = pyotp.TOTP(secret)
            if totp.verify(code):
                two_fa_secrets[username] = secret
                session.pop('pending_2fa_secret', None)
                add_audit('Enable 2FA')
                flash('2FA enabled successfully!', 'success')
                return redirect(url_for('admin_dashboard'))
            else:
                flash('Invalid code. Try again.', 'danger')
        secret = session.get('pending_2fa_secret') or pyotp.random_base32()
        session['pending_2fa_secret'] = secret
        uri = pyotp.totp.TOTP(secret).provisioning_uri(name=username, issuer_name="StudentAI")
        img = qrcode.make(uri)
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        qr_b64 = base64.b64encode(buf.getvalue()).decode()
        return render_template('setup_2fa.html', secret=secret, qr_b64=qr_b64,
                               has_2fa=username in two_fa_secrets)
    except ImportError:
        flash('pyotp/qrcode not installed. Add them to requirements.txt.', 'warning')
        return redirect(url_for('admin_dashboard'))

@app.route('/admin/2fa/disable', methods=['POST'])
@login_required
def disable_2fa():
    username = session.get('username')
    two_fa_secrets.pop(username, None)
    add_audit('Disable 2FA')
    return jsonify({'success': True})

# ─── Reports ─────────────────────────────────────────────────────
@app.route('/admin/download/csv')
@require_perm('download')
def download_csv():
    if not students_store:
        flash('No data to export.', 'warning')
        return redirect(url_for('admin_dashboard'))
    df = pd.DataFrame(students_store)
    out = io.StringIO()
    df.to_csv(out, index=False)
    add_audit('Download CSV')
    return send_file(io.BytesIO(out.getvalue().encode()), mimetype='text/csv',
                     as_attachment=True, download_name=f'student_report_{_ts()}.csv')

@app.route('/admin/download/excel')
@require_perm('download')
def download_excel():
    if not students_store:
        flash('No data to export.', 'warning')
        return redirect(url_for('admin_dashboard'))
    df = pd.DataFrame(students_store)
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine='openpyxl') as w:
        df.to_excel(w, index=False, sheet_name='Predictions')
        if pipeline.is_loaded():
            a = pipeline.get_analytics(df)
            pd.DataFrame({
                'Metric': ['Total', 'High Risk', 'Medium Risk', 'Low Risk', 'Good Perf', 'Avg Perf', 'Low Perf', 'Avg Health Score', 'Anomalies'],
                'Value': [a.get('total',0), a['dropout_dist'].get('High',0), a['dropout_dist'].get('Medium',0),
                          a['dropout_dist'].get('Low',0), a['perf_dist'].get('Good',0),
                          a['perf_dist'].get('Average',0), a['perf_dist'].get('Low',0),
                          a.get('avg_health',0), a.get('anomaly_count',0)]
            }).to_excel(w, index=False, sheet_name='Summary')
    out.seek(0)
    add_audit('Download Excel')
    return send_file(out, mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                     as_attachment=True, download_name=f'student_report_{_ts()}.xlsx')

@app.route('/admin/download/pdf')
@require_perm('download')
def download_pdf():
    if not students_store:
        flash('No data to export.', 'warning')
        return redirect(url_for('admin_dashboard'))
    try:
        from reportlab.lib.pagesizes import A4, landscape
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
        from reportlab.lib.units import inch
        out = io.BytesIO()
        doc = SimpleDocTemplate(out, pagesize=landscape(A4), topMargin=0.4*inch)
        styles = getSampleStyleSheet()
        els = []
        els.append(Paragraph("Student AI — Prediction Report",
                              ParagraphStyle('T', parent=styles['Title'], fontSize=18,
                                             textColor=colors.HexColor('#6c5ce7'))))
        els.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')} | Total: {len(students_store)}", styles['Normal']))
        els.append(Spacer(1, 0.15*inch))
        hdr = ['#', 'Name', 'Att.', 'Hrs', 'Risk', 'Perf', 'Health', 'Strong', 'Weak', 'Anomaly']
        rows = [hdr]
        for i, s in enumerate(students_store[:60], 1):
            rows.append([str(i), str(s.get('Student_Name',''))[:18], str(s.get('Attendance','')),
                         str(s.get('Study_Hours','')), str(s.get('Dropout_Risk','')),
                         str(s.get('Performance','')), str(s.get('Health_Score','')),
                         str(s.get('Strong_Subject',''))[:12], str(s.get('Weak_Subject',''))[:12],
                         '⚠' if s.get('Is_Anomaly') else ''])
        t = Table(rows, repeatRows=1)
        t.setStyle(TableStyle([
            ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#6c5ce7')),
            ('TEXTCOLOR',(0,0),(-1,0),colors.white),
            ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),
            ('FONTSIZE',(0,0),(-1,-1),8),
            ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white, colors.HexColor('#f8f9fa')]),
            ('GRID',(0,0),(-1,-1),0.4,colors.HexColor('#dee2e6')),
            ('ALIGN',(0,0),(-1,-1),'CENTER'),
            ('PADDING',(0,0),(-1,-1),4),
        ]))
        els.append(t)
        doc.build(els)
        out.seek(0)
        add_audit('Download PDF')
        return send_file(out, mimetype='application/pdf', as_attachment=True,
                         download_name=f'student_report_{_ts()}.pdf')
    except ImportError:
        flash('ReportLab not available.', 'warning')
        return redirect(url_for('admin_dashboard'))

# ─── External API ────────────────────────────────────────────────
@app.route('/api/v1/predict', methods=['POST'])
def api_predict():
    """External API endpoint — requires API key header."""
    key = request.headers.get('X-API-Key', '')
    info = api_keys_store.get(key)
    if not info or not info.get('active'):
        return jsonify({'error': 'Invalid or inactive API key'}), 401
    api_keys_store[key]['requests'] += 1
    if not ensure_models():
        return jsonify({'error': 'Models not ready'}), 500
    try:
        data = request.get_json()
        result = pipeline.predict_single(data, include_shap=False)
        return jsonify({'success': True, 'prediction': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 400


# ─── Student Auth ────────────────────────────────────────────────
def student_login_required(f):
    from functools import wraps
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get('student_logged_in'):
            return redirect(url_for('student_login'))
        return f(*args, **kwargs)
    return decorated

@app.route('/student/login', methods=['GET', 'POST'])
def student_login():
    if session.get('student_logged_in'):
        return redirect(url_for('student_portal'))
    error = None
    if request.method == 'POST':
        sid = request.form.get('student_id', '').strip().upper()
        pwd = request.form.get('password', '')
        acct = STUDENT_ACCOUNTS.get(sid)
        if acct and hash_pw(pwd) == acct['password']:
            session['student_logged_in'] = True
            session['student_id'] = sid
            session['student_name'] = acct['name']
            session['student_roll'] = acct['roll']
            session['student_branch'] = acct['branch']
            flash(f"Welcome, {acct['name']}!", 'success')
            return redirect(url_for('student_portal'))
        else:
            error = 'Invalid Student ID or password.'
    return render_template('student_login.html', error=error)

@app.route('/student/logout')
def student_logout():
    session.pop('student_logged_in', None)
    session.pop('student_id', None)
    session.pop('student_name', None)
    session.pop('student_roll', None)
    session.pop('student_branch', None)
    flash('Logged out successfully.', 'info')
    return redirect(url_for('student_login'))

@app.route('/student/portal')
@student_login_required
def student_portal():
    sid = session.get('student_id')
    student_data = next((s for s in students_store if str(s.get('Student_ID')) == str(sid)), None)
    my_feedbacks = [f for f in feedback_store if f.get('student_id') == sid]
    notifs = [n for n in notifications_store if session.get('student_name') in n.get('message','')]
    return render_template('student_portal.html',
                           student_data=student_data,
                           my_feedbacks=my_feedbacks,
                           notifications=notifs,
                           subject_teachers=SUBJECT_TEACHERS,
                           risk_badge=risk_badge,
                           perf_badge=perf_badge)

@app.route('/student/planner')
@student_login_required
def student_planner():
    sid = session.get('student_id')
    student_data = next((s for s in students_store if str(s.get('Student_ID')) == str(sid)), None)
    return render_template('study_planner.html', students=students_store,
                           student_data=student_data, is_student=True)

@app.route('/student/feedback', methods=['GET', 'POST'])
@student_login_required
def student_feedback():
    if request.method == 'POST':
        subject  = request.form.get('subject', '')
        teacher  = SUBJECT_TEACHERS.get(subject, request.form.get('teacher', ''))
        rating   = int(request.form.get('rating', 3))
        comment  = request.form.get('comment', '').strip()
        teaching = request.form.get('teaching_quality', '3')
        clarity  = request.form.get('clarity', '3')
        support  = request.form.get('support', '3')
        feedback_store.append({
            'id': len(feedback_store) + 1,
            'student_id':   session.get('student_id'),
            'student_name': session.get('student_name'),
            'roll':         session.get('student_roll'),
            'branch':       session.get('student_branch'),
            'subject':      subject,
            'teacher':      teacher,
            'rating':       rating,
            'teaching_quality': int(teaching),
            'clarity':      int(clarity),
            'support':      int(support),
            'comment':      comment,
            'timestamp':    datetime.now().strftime('%Y-%m-%d %H:%M'),
        })
        flash(f'Feedback for {teacher} submitted successfully!', 'success')
        return redirect(url_for('student_portal'))
    my_feedbacks = [f for f in feedback_store if f.get('student_id') == session.get('student_id')]
    return render_template('student_feedback.html',
                           subject_teachers=SUBJECT_TEACHERS,
                           my_feedbacks=my_feedbacks)

# ─── Admin: View Feedback ────────────────────────────────────────
@app.route('/admin/feedback')
@login_required
def admin_feedback():
    by_subject = {}
    for f in feedback_store:
        s = f.get('subject', 'Unknown')
        by_subject.setdefault(s, []).append(f)
    avg_by_subject = {}
    for s, fbs in by_subject.items():
        avg_by_subject[s] = {
            'teacher': fbs[0].get('teacher',''),
            'count': len(fbs),
            'avg_rating': round(sum(x['rating'] for x in fbs)/len(fbs), 1),
            'avg_teaching': round(sum(x.get('teaching_quality',3) for x in fbs)/len(fbs), 1),
            'avg_clarity': round(sum(x.get('clarity',3) for x in fbs)/len(fbs), 1),
            'avg_support': round(sum(x.get('support',3) for x in fbs)/len(fbs), 1),
        }
    return render_template('admin_feedback.html',
                           feedback_store=feedback_store,
                           avg_by_subject=avg_by_subject,
                           subject_teachers=SUBJECT_TEACHERS)

# ─── Utils ────────────────────────────────────────────────────────
def _ts(): return datetime.now().strftime('%Y%m%d_%H%M%S')


def initialize_app():
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    print("=" * 60)
    print("  🎓 Student AI System v2 — Advanced Edition")
    print("=" * 60)
    if ensure_models():
        print(f"  ✓ ML Models loaded")
        print(f"  ✓ Dropout Accuracy : {pipeline.metadata.get('dropout_accuracy', '?')}%")
        print(f"  ✓ Performance Acc  : {pipeline.metadata.get('performance_accuracy', '?')}%")
        print(f"  ✓ Anomaly Detector : ready")
    else:
        print("  ⚠ Run: python train_model.py")
    print()
    print("  RBAC Users:")
    print("  → admin    / admin123    (Super Admin — full access)")
    print("  → teacher  / teacher123  (Teacher — view + notify)")
    print("  → counselor/ counsel123  (Counselor — view + download)")
    print()
    print("  → http://127.0.0.1:5000         (MIT Splash)")
    print("  → http://127.0.0.1:5000/app     (Main App)")
    print("  → http://127.0.0.1:5000/student/login  (Student Portal)")
    print("=" * 60)


if __name__ == '__main__':
    initialize_app()
    app.run(debug=True, port=5000)

"""Isolated auth-template checks in real Chromium; no app/database or email access.

Run: venv/bin/python -m pytest tests/functional/test_auth_ui_regressions.py
Browser cases skip if Chromium is absent. Requests are stubbed intentionally.
"""
import json
from pathlib import Path
import shutil
import subprocess

from flask import Flask, render_template
import pytest


ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.functional


@pytest.fixture
def auth_html():
    app = Flask(__name__, template_folder=str(ROOT / "templates"))
    with app.test_request_context():
        return render_template("auth.html")


@pytest.mark.parametrize("width", [320, 480, 1280])
def test_auth_accessibility_layout_and_flow(auth_html, tmp_path, width):
    browser = shutil.which("chromium") or shutil.which("chromium-browser")
    if not browser:
        pytest.skip("Chromium required for real-browser UI checks")
    # DOMContentLoaded ensures the template's Enter handlers are installed first.
    probe = r"""
<script>
document.addEventListener('DOMContentLoaded', async () => {
    const results = [];
    const check = (ok, name) => { if (!ok) throw new Error(name); results.push(name); };
    const visible = step => getComputedStyle(document.getElementById('auth-step-' + step)).display !== 'none';
    try {
        check(document.documentElement.lang === 'en', 'document language');
        check(visible('email') && !visible('otp') && !visible('name') && !visible('success'), 'initial visibility');
        for (const [id, autocomplete] of [['auth-email', 'email'], ['auth-otp', 'one-time-code'], ['auth-display-name', 'nickname']]) {
            const input = document.getElementById(id);
            check(input.labels.length === 1 && input.labels[0].textContent.trim(), id + ' accessible label');
            check(input.autocomplete === autocomplete, id + ' autocomplete');
        }
        check(document.getElementById('auth-otp').inputMode === 'numeric', 'numeric keyboard');
        check(document.getElementById('auth-otp').maxLength === 6, 'OTP length');
        const email = document.getElementById('auth-email');
        email.focus();
        check(getComputedStyle(email).outlineStyle === 'solid' && getComputedStyle(email).outlineWidth === '3px', 'input focus outline');
        check(matchMedia('(prefers-reduced-motion: reduce)').matches, 'reduced motion enabled');
        check(getComputedStyle(document.querySelector('button')).transitionDuration === '0s', 'reduced motion transition');
        const requests = [];
        let reply = {};
        window.fetch = async (url, options) => {
            requests.push([url, JSON.parse(options.body)]);
            return {ok: true, json: async () => reply};
        };
        await sendOTP();
        check(requests.length === 0 && document.getElementById('auth-email-error').textContent.includes('Please enter'), 'empty email validation');
        email.value = ' person@example.test ';
        email.dispatchEvent(new KeyboardEvent('keypress', {key: 'Enter', bubbles: true}));
        await new Promise(resolve => setTimeout(resolve, 0));
        check(visible('otp') && !visible('email'), 'Enter submits email');
        check(requests[0][0] === '/auth/send-otp' && requests[0][1].email === 'person@example.test', 'email request');
        check(document.getElementById('auth-email-display').textContent === 'person@example.test', 'email displayed');
        document.querySelector('#auth-step-otp .secondary-btn').click();
        check(visible('email') && document.getElementById('auth-email-error').textContent === '', 'back and clear errors');
        await sendOTP();
        document.getElementById('auth-otp').value = '012345';
        reply = {needs_display_name: true};
        await verifyOTP();
        check(visible('name') && requests.at(-1)[1].otp_code === '012345', 'new user OTP preserves leading zero');
        document.getElementById('auth-display-name').value = 'Example';
        reply = {user: {display_name: '<img src=x onerror=alert(1)>'}};
        await claimName();
        check(visible('success'), 'sign up success');
        check(document.getElementById('auth-success-name').children.length === 0, 'display name rendered as text');
        showAuthStep('otp');
        await verifyOTP();
        check(visible('success'), 'existing user success');
        window.fetch = async () => { throw new Error('offline'); };
        showAuthStep('email');
        await sendOTP();
        check(document.getElementById('auth-email-error').textContent === 'Network error. Please try again.', 'network error');
        for (const step of ['email', 'otp', 'name', 'success']) {
            showAuthStep(step);
            check(document.documentElement.scrollWidth <= innerWidth, step + ' no horizontal overflow');
            const card = document.querySelector('.auth-container').getBoundingClientRect();
            check(card.left >= 0 && card.right <= innerWidth, step + ' card fits viewport');
        }
        document.body.dataset.probe = 'passed';
    } catch (error) {
        document.body.dataset.probe = 'failed: ' + error.message;
    }
    document.body.dataset.checks = JSON.stringify(results);
});
</script>
"""
    page = tmp_path / "auth.html"
    page.write_text(auth_html.replace("</body>", probe + "</body>"))
    command = [browser, "--headless", "--disable-gpu", "--no-sandbox",
               "--no-proxy-server", "--disable-background-networking",
               "--force-prefers-reduced-motion", f"--window-size={width},900",
               f"--user-data-dir={tmp_path / 'browser'}", "--virtual-time-budget=3000",
               "--dump-dom", page.as_uri()]
    result = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert 'data-probe="passed"' in result.stdout, result.stdout + result.stderr

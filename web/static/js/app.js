/**
 * Scholarship Eligibility Predictor - Client Application Script
 * Connects frontend form to the Flask ML backend /api/predict.
 */

document.addEventListener('DOMContentLoaded', () => {
  // Elements
  const form = document.getElementById('prediction-form');
  const submitBtn = document.getElementById('submit-btn');
  const btnSpinner = document.getElementById('btn-spinner');
  const formError = document.getElementById('form-error');
  const errorMessage = document.getElementById('error-message');
  const resultContainer = document.getElementById('result-container');
  const resultBadge = document.getElementById('result-badge');
  const resultIcon = document.getElementById('result-icon');
  const resultTitle = document.getElementById('result-title');
  const confidenceValue = document.getElementById('confidence-value');
  const eligibleProbValue = document.getElementById('eligible-prob-value');
  const notEligibleProbValue = document.getElementById('not-eligible-prob-value');
  const eligibleProgress = document.getElementById('eligible-progress');
  const notEligibleProgress = document.getElementById('not-eligible-progress');
  const factorsList = document.getElementById('factors-list');
  const summaryGrid = document.getElementById('summary-grid');
  const incomeInput = document.getElementById('familyIncome');
  const incomeFormatted = document.getElementById('income-formatted');

  // Quick Demo Buttons
  const loadEligibleBtn = document.getElementById('load-eligible-btn');
  const loadIneligibleBtn = document.getElementById('load-ineligible-btn');
  const resetFormBtn = document.getElementById('reset-form-btn');
  const checkAnotherBtn = document.getElementById('check-another-btn');

  // Status Pill
  const statusPill = document.getElementById('system-status-pill');
  const statusText = document.getElementById('status-text');

  // Modal elements
  const modal = document.getElementById('image-modal');
  const modalImg = document.getElementById('modal-img');
  const modalCaption = document.getElementById('modal-caption');
  const modalClose = document.getElementById('modal-close');

  // =========================================================================
  // 1. Health Check at Initialization
  // =========================================================================
  async function checkSystemHealth() {
    try {
      const response = await fetch('/api/health');
      if (response.ok) {
        const data = await response.json();
        statusPill.className = 'system-status-pill online';
        statusText.textContent = `Model Active (RF ${data.test_accuracy})`;
      } else {
        throw new Error('Server returned error status');
      }
    } catch (err) {
      statusPill.className = 'system-status-pill error';
      statusText.textContent = 'Server Offline';
      console.warn('Backend health check error:', err);
    }
  }

  checkSystemHealth();

  // =========================================================================
  // 2. INR Live Formatting Helper
  // =========================================================================
  function formatINR(val) {
    if (!val || isNaN(val)) return '';
    const num = Number(val);
    return '₹' + num.toLocaleString('en-IN');
  }

  incomeInput.addEventListener('input', () => {
    const val = incomeInput.value;
    if (val && !isNaN(val) && Number(val) >= 0) {
      incomeFormatted.textContent = formatINR(val);
      incomeFormatted.classList.add('visible');
    } else {
      incomeFormatted.classList.remove('visible');
    }
  });

  // =========================================================================
  // 3. Quick Demo Preset Loaders
  // =========================================================================
  const DEMO_PROFILES = {
    eligible: {
      Gender: 'Female',
      Community: 'SC',
      FamilyIncome: 120000,
      '12thMarks': 85.0,
      FirstGraduate: 'Yes',
      District: 'Chennai',
      CollegeType: 'Government',
      Course: 'Engineering'
    },
    ineligible: {
      Gender: 'Male',
      Community: 'OC',
      FamilyIncome: 650000,
      '12thMarks': 55.0,
      FirstGraduate: 'No',
      District: 'Chennai',
      CollegeType: 'Private',
      Course: 'Management'
    }
  };

  function applyProfile(profile) {
    document.getElementById('gender').value = profile.Gender;
    document.getElementById('community').value = profile.Community;
    document.getElementById('familyIncome').value = profile.FamilyIncome;
    document.getElementById('twelfthMarks').value = profile['12thMarks'];
    document.getElementById('firstGraduate').value = profile.FirstGraduate;
    document.getElementById('district').value = profile.District;
    document.getElementById('collegeType').value = profile.CollegeType;
    document.getElementById('course').value = profile.Course;

    incomeFormatted.textContent = formatINR(profile.FamilyIncome);
    incomeFormatted.classList.add('visible');
    hideError();
  }

  loadEligibleBtn.addEventListener('click', () => applyProfile(DEMO_PROFILES.eligible));
  loadIneligibleBtn.addEventListener('click', () => applyProfile(DEMO_PROFILES.ineligible));

  function resetForm() {
    form.reset();
    incomeFormatted.classList.remove('visible');
    hideError();
    resultContainer.classList.add('hidden');
  }

  resetFormBtn.addEventListener('click', resetForm);
  checkAnotherBtn.addEventListener('click', () => {
    document.getElementById('predictor').scrollIntoView({ behavior: 'smooth' });
    document.getElementById('familyIncome').focus();
  });

  // =========================================================================
  // 4. Error Display Helpers
  // =========================================================================
  function showError(msg) {
    errorMessage.textContent = msg;
    formError.classList.remove('hidden');
    formError.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }

  function hideError() {
    formError.classList.add('hidden');
    errorMessage.textContent = '';
  }

  // =========================================================================
  // 5. Form Submission & Inference API Call
  // =========================================================================
  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    hideError();

    // Client-side extraction
    const gender = document.getElementById('gender').value;
    const community = document.getElementById('community').value;
    const incomeStr = document.getElementById('familyIncome').value;
    const marksStr = document.getElementById('twelfthMarks').value;
    const firstGraduate = document.getElementById('firstGraduate').value;
    const district = document.getElementById('district').value;
    const collegeType = document.getElementById('collegeType').value;
    const course = document.getElementById('course').value;

    // Field presence checks
    if (!gender || !community || !incomeStr || !marksStr || !firstGraduate || !district || !collegeType || !course) {
      showError('Please complete all 8 required student attributes before submitting.');
      return;
    }

    const income = parseFloat(incomeStr);
    if (isNaN(income) || income < 0) {
      showError('Please enter a valid non-negative Annual Family Income in INR.');
      return;
    }

    const marks = parseFloat(marksStr);
    if (isNaN(marks) || marks < 0 || marks > 100) {
      showError('12th Board Marks must be a valid percentage between 0.0% and 100.0%.');
      return;
    }

    const payload = {
      Gender: gender,
      Community: community,
      FamilyIncome: income,
      '12thMarks': marks,
      FirstGraduate: firstGraduate,
      District: district,
      CollegeType: collegeType,
      Course: course
    };

    // UI Loading State
    submitBtn.disabled = true;
    btnSpinner.classList.remove('hidden');
    document.querySelector('.btn-text').textContent = 'Evaluating Decision Trees...';

    try {
      const response = await fetch('/api/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json'
        },
        body: JSON.stringify(payload)
      });

      const data = await response.json();

      if (!response.ok || data.status !== 'success') {
        throw new Error(data.message || 'An error occurred during evaluation. Please check your inputs.');
      }

      displayResult(data);

    } catch (err) {
      showError(err.message || 'Unable to connect to the prediction server. Please try again.');
    } finally {
      submitBtn.disabled = false;
      btnSpinner.classList.add('hidden');
      document.querySelector('.btn-text').textContent = 'Check Eligibility';
    }
  });

  // =========================================================================
  // 6. Result Card Renderer
  // =========================================================================
  function displayResult(data) {
    const isEligible = data.is_eligible;

    // Card Theme & Badges
    resultContainer.className = 'result-card ' + (isEligible ? 'eligible' : 'ineligible');
    resultBadge.className = 'result-badge ' + (isEligible ? 'eligible' : 'ineligible');

    if (isEligible) {
      resultIcon.innerHTML = '&#10003;'; // Checkmark
      resultTitle.textContent = 'ELIGIBLE FOR SCHOLARSHIP SCREENING';
    } else {
      resultIcon.innerHTML = '&#10005;'; // Cross mark
      resultTitle.textContent = 'NOT ELIGIBLE FOR SCHOLARSHIP SCREENING';
    }

    // Probability & Confidence Metrics
    confidenceValue.textContent = `${data.confidence}%`;
    eligibleProbValue.textContent = `${data.eligible_probability}%`;
    notEligibleProbValue.textContent = `${data.not_eligible_probability}%`;

    // Dynamic Progress Bar Animations
    setTimeout(() => {
      eligibleProgress.style.width = `${data.eligible_probability}%`;
      notEligibleProgress.style.width = `${data.not_eligible_probability}%`;
    }, 50);

    // Decision Factors
    factorsList.innerHTML = '';
    if (data.decision_factors && data.decision_factors.length > 0) {
      data.decision_factors.forEach(factor => {
        const li = document.createElement('li');
        li.textContent = factor;
        factorsList.appendChild(li);
      });
    } else {
      const li = document.createElement('li');
      li.textContent = 'Multi-feature threshold interactions determined final classification split.';
      factorsList.appendChild(li);
    }

    // Evaluated Profile Summary Grid
    const summary = data.input_summary;
    summaryGrid.innerHTML = `
      <div class="summary-item">
        <span class="summary-item-label">Gender</span>
        <span class="summary-item-value">${summary.Gender}</span>
      </div>
      <div class="summary-item">
        <span class="summary-item-label">Community</span>
        <span class="summary-item-value">${summary.Community}</span>
      </div>
      <div class="summary-item">
        <span class="summary-item-label">Family Income</span>
        <span class="summary-item-value">${formatINR(summary.FamilyIncome)}</span>
      </div>
      <div class="summary-item">
        <span class="summary-item-label">12th Marks</span>
        <span class="summary-item-value">${summary['12thMarks']}%</span>
      </div>
      <div class="summary-item">
        <span class="summary-item-label">First Graduate</span>
        <span class="summary-item-value">${summary.FirstGraduate}</span>
      </div>
      <div class="summary-item">
        <span class="summary-item-label">District</span>
        <span class="summary-item-value">${summary.District}</span>
      </div>
      <div class="summary-item">
        <span class="summary-item-label">College Type</span>
        <span class="summary-item-value">${summary.CollegeType}</span>
      </div>
      <div class="summary-item">
        <span class="summary-item-label">Course</span>
        <span class="summary-item-value">${summary.Course}</span>
      </div>
    `;

    // Reveal and scroll smoothly
    resultContainer.classList.remove('hidden');
    resultContainer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }

  // =========================================================================
  // 7. Lightbox Modal for Visual Dashboard Plots
  // =========================================================================
  const plotCards = document.querySelectorAll('.plot-card');
  plotCards.forEach(card => {
    const wrapper = card.querySelector('.plot-img-wrapper');
    const img = card.querySelector('.plot-img');
    const title = card.querySelector('.plot-title')?.textContent || '';
    const desc = card.querySelector('.plot-desc')?.textContent || '';

    wrapper.addEventListener('click', () => {
      modalImg.src = img.src;
      modalCaption.innerHTML = `<strong>${title}</strong> &mdash; ${desc}`;
      modal.classList.remove('hidden');
    });
  });

  function closeModal() {
    modal.classList.add('hidden');
    modalImg.src = '';
  }

  modalClose.addEventListener('click', closeModal);
  modal.querySelector('.modal-backdrop').addEventListener('click', closeModal);
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && !modal.classList.contains('hidden')) {
      closeModal();
    }
  });
});

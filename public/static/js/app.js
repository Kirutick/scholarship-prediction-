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
  const recommendationsList = document.getElementById('recommendations-list');
  const scholarshipMatchesList = document.getElementById('scholarship-matches-list');
  const catalogWarning = document.getElementById('catalog-warning');
  const estimatedSupportValue = document.getElementById('estimated-support');
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
  const catalogGrid = document.getElementById('catalog-grid');
  const catalogSearch = document.getElementById('scholarship-search');
  const openOnlyFilter = document.getElementById('open-only-filter');
  const catalogEmpty = document.getElementById('catalog-empty');
  const catalogUpdated = document.getElementById('catalog-updated');
  const savedList = document.getElementById('saved-scholarships-list');
  const savedEmpty = document.getElementById('saved-empty');
  const compareSection = document.getElementById('compare-section');
  const compareTableBody = document.getElementById('compare-table-body');

  const PLANNER_STEPS = [
    'Review the official eligibility guidelines',
    'Confirm the current application window',
    'Check the official document requirements',
    'Open the official application portal'
  ];
  const APPLICATION_STATUSES = [
    'Not Started', 'Documents Collecting', 'Ready to Apply',
    'Applied', 'Verification Pending', 'Approved', 'Rejected'
  ];
  const STORAGE_KEY = 'scholarshipAssistantPlannerV1';
  let plannerState = loadPlannerState();
  let catalogRecords = [];
  const catalogRecordById = new Map();
  let searchTimer;

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
  loadScholarships();

  // =========================================================================
  // 2. INR Live Formatting Helper
  // =========================================================================
  function formatINR(val) {
    if (!val || isNaN(val)) return '';
    const num = Number(val);
    return '₹' + num.toLocaleString('en-IN');
  }

  function loadPlannerState() {
    try {
      const stored = localStorage.getItem(STORAGE_KEY);
      if (!stored) return { saved: [], compare: [], entries: {} };
      const parsed = JSON.parse(stored);
      return {
        saved: Array.isArray(parsed.saved) ? parsed.saved.filter(id => typeof id === 'string') : [],
        compare: Array.isArray(parsed.compare) ? parsed.compare.filter(id => typeof id === 'string').slice(0, 3) : [],
        entries: parsed.entries && typeof parsed.entries === 'object' ? parsed.entries : {}
      };
    } catch (err) {
      console.error('Could not read locally saved scholarship planner data:', err);
      return { saved: [], compare: [], entries: {} };
    }
  }

  function savePlannerState() {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(plannerState));
    } catch (err) {
      console.error('Could not save scholarship planner data locally:', err);
      catalogWarning.textContent = 'Browser storage is unavailable. Your saved list could not be stored.';
      catalogWarning.classList.remove('hidden');
    }
  }

  function formatDate(value) {
    if (!value) return 'Date not available';
    const date = new Date(`${value}T00:00:00`);
    return Number.isNaN(date.getTime())
      ? 'Date not available'
      : new Intl.DateTimeFormat('en-IN', { day: 'numeric', month: 'short', year: 'numeric' }).format(date);
  }

  function textElement(tagName, className, text) {
    const element = document.createElement(tagName);
    if (className) element.className = className;
    element.textContent = text || '';
    return element;
  }

  function createOfficialLink(label, href) {
    if (!href) return textElement('span', 'unknown-value', 'Not available in source');
    const link = textElement('a', 'official-link', label);
    link.href = href;
    link.target = '_blank';
    link.rel = 'noopener noreferrer';
    return link;
  }

  function toggleSaved(id) {
    plannerState.saved = plannerState.saved.includes(id)
      ? plannerState.saved.filter(savedId => savedId !== id)
      : [...plannerState.saved, id];
    if (!plannerState.entries[id]) {
      plannerState.entries[id] = { status: 'Not Started', steps: {} };
    }
    savePlannerState();
    renderCatalogCards(catalogRecords);
    renderSavedPlanner();
  }

  function toggleCompare(id, checked) {
    if (checked && !plannerState.compare.includes(id)) {
      if (plannerState.compare.length >= 3) {
        catalogWarning.textContent = 'Compare up to three scholarships at a time.';
        catalogWarning.classList.remove('hidden');
        renderCatalogCards(catalogRecords);
        return;
      }
      plannerState.compare.push(id);
    } else if (!checked) {
      plannerState.compare = plannerState.compare.filter(compareId => compareId !== id);
    }
    savePlannerState();
    renderCatalogCards(catalogRecords);
    renderSavedPlanner();
  }

  function createScholarshipCard(record, view = 'catalog') {
    const card = document.createElement('article');
    card.className = 'catalog-card';
    card.dataset.scholarshipId = record.id;
    card.appendChild(textElement('h3', 'catalog-card-title', record.name));

    const badges = document.createElement('div');
    badges.className = 'catalog-badges';
    badges.append(
      textElement('span', 'catalog-badge status-badge', record.status || 'Needs Verification'),
      textElement('span', 'catalog-badge source-badge', record.source_status === 'official'
        ? 'Verified official information'
        : 'Official source · partial details')
    );
    card.appendChild(badges);

    const score = record.match_score === null || record.match_score === undefined
      ? 'Match score unavailable — eligibility rules are incomplete'
      : `${record.match_score}/100 compatibility with documented criteria`;
    card.appendChild(textElement('p', 'catalog-score', score));
    card.appendChild(textElement('p', 'catalog-provider', `${record.provider} · ${record.category}`));
    card.appendChild(textElement(
      'p',
      'catalog-deadline',
      `${record.deadline_status || 'DATE NOT AVAILABLE'} · Student closing date: ${formatDate(record.deadline)}`
    ));

    const actions = document.createElement('div');
    actions.className = 'catalog-actions';
    const saved = plannerState.saved.includes(record.id);
    const saveButton = textElement('button', 'btn btn-sm btn-outline', saved ? '♥ Saved' : '♡ Save');
    saveButton.type = 'button';
    saveButton.setAttribute('aria-pressed', String(saved));
    saveButton.addEventListener('click', () => toggleSaved(record.id));
    actions.appendChild(saveButton);
    if (view === 'catalog') {
      const compareLabel = document.createElement('label');
      compareLabel.className = 'compare-checkbox';
      const compareInput = document.createElement('input');
      compareInput.type = 'checkbox';
      compareInput.checked = plannerState.compare.includes(record.id);
      compareInput.addEventListener('change', () => toggleCompare(record.id, compareInput.checked));
      compareLabel.append(compareInput, document.createTextNode('Compare'));
      actions.appendChild(compareLabel);
    }
    card.appendChild(actions);

    const details = document.createElement('details');
    details.className = 'scholarship-details';
    details.appendChild(textElement('summary', '', 'Eligibility, benefits, documents and sources'));
    const detailGrid = document.createElement('div');
    detailGrid.className = 'scholarship-detail-grid';
    detailGrid.append(
      textElement('div', 'detail-value', `Why this appears: ${(record.reasons || []).join(' ')}`),
      textElement('div', 'detail-value', `Matched documented rules: ${(record.matched_rules || []).join('; ') || 'None available in the source snapshot.'}`),
      textElement('div', 'detail-value', `Known rule conflicts: ${(record.failed_rules || []).join('; ') || 'None established from the available criteria.'}`),
      textElement('div', 'detail-value', `Information to confirm: ${(record.missing_information || []).join(', ') || 'See the official guideline; the source record is partial.'}`),
      textElement('div', 'detail-value', `Known but unverified criteria: ${(record.unverified_criteria || []).join(', ') || 'None listed.'}`),
      textElement('div', 'detail-value', `Benefit: ${record.benefits?.display || 'Not specified in the verified source.'}`),
      textElement('div', 'detail-value', `Application window: ${record.application_window || 'Not specified'}`),
      textElement('div', 'detail-value', `Institute verification: ${formatDate(record.verification_deadlines?.institution)} · L2 verification: ${formatDate(record.verification_deadlines?.level_2)}`),
      textElement('div', 'detail-value', `Required documents: ${record.documents_status === 'verified' ? (record.documents_display || []).join(', ') : 'Not specified in the verified source.'}`),
      textElement('div', 'detail-value', `Information last checked: ${formatDate(record.last_verified)}`)
    );

    const source = record.source || {};
    const sourceNote = document.createElement('p');
    sourceNote.className = 'detail-value source-note';
    sourceNote.textContent = source.limitations || 'Review the linked official source for current terms.';
    detailGrid.appendChild(sourceNote);
    if (Array.isArray(source.verified_facts)) {
      const verifiedFacts = document.createElement('ul');
      verifiedFacts.className = 'verified-facts';
      source.verified_facts.forEach(fact => verifiedFacts.appendChild(textElement('li', '', fact)));
      detailGrid.appendChild(verifiedFacts);
    }
    const links = document.createElement('div');
    links.className = 'catalog-links';
    links.appendChild(createOfficialLink('View Official Details', record.official_source_url));
    if (record.application_url) links.appendChild(createOfficialLink('Open Official Application Portal', record.application_url));
    detailGrid.appendChild(links);
    detailGrid.appendChild(textElement('p', 'detail-disclaimer', record.disclaimer || 'Always verify current official guidelines before applying.'));
    details.appendChild(detailGrid);
    card.appendChild(details);
    return card;
  }

  function renderCatalogCards(records) {
    catalogGrid.replaceChildren();
    records.forEach(record => catalogGrid.appendChild(createScholarshipCard(record)));
    catalogEmpty.classList.toggle('hidden', records.length > 0);
    if (records.length === 0 && !catalogEmpty.textContent) {
      catalogEmpty.textContent = 'No catalog entries match these search filters.';
    }
  }

  async function loadScholarships() {
    const params = new URLSearchParams();
    if (catalogSearch.value.trim()) params.set('q', catalogSearch.value.trim());
    if (openOnlyFilter.checked) params.set('open', 'true');
    try {
      const response = await fetch(`/api/scholarships?${params.toString()}`, { cache: 'no-store' });
      const result = await response.json();
      if (!response.ok || result.status !== 'success') {
        throw new Error(result.message || 'Scholarship catalog could not be loaded.');
      }
      catalogRecords = result.scholarships;
      catalogRecords.forEach(record => catalogRecordById.set(record.id, record));
      catalogEmpty.textContent = 'No catalog entries match these search filters.';
      renderCatalogCards(catalogRecords);
      catalogUpdated.textContent = `Reviewed catalog updated ${formatDate(result.database_last_updated)} · ${result.count} records`;
      catalogWarning.classList.add('hidden');
      renderSavedPlanner();
    } catch (err) {
      console.error('Scholarship catalog request failed:', err);
      catalogGrid.replaceChildren();
      catalogEmpty.textContent = 'Scholarship source data is temporarily unavailable. Eligibility screening remains available.';
      catalogEmpty.classList.remove('hidden');
      catalogUpdated.textContent = 'Source verification unavailable.';
    }
  }

  catalogSearch.addEventListener('input', () => {
    clearTimeout(searchTimer);
    searchTimer = setTimeout(loadScholarships, 250);
  });
  openOnlyFilter.addEventListener('change', loadScholarships);

  function renderMatches(records) {
    scholarshipMatchesList.replaceChildren();
    if (!Array.isArray(records) || records.length === 0) {
      scholarshipMatchesList.appendChild(textElement(
        'li',
        'catalog-empty-inline',
        'No source-linked records are available for this profile. Browse the catalog or check again after its official rules are updated.'
      ));
      return;
    }
    records.forEach(record => {
      catalogRecordById.set(record.id, record);
      scholarshipMatchesList.appendChild(createScholarshipCard(record, 'match'));
    });
  }

  function updateCompareTable() {
    const selected = plannerState.compare
      .map(id => catalogRecordById.get(id))
      .filter(Boolean);
    compareTableBody.replaceChildren();
    selected.forEach(record => {
      const row = document.createElement('tr');
      [
        record.name,
        record.match_score === null ? 'Not available' : `${record.match_score}/100`,
        record.benefits?.display || 'Not specified',
        `${record.deadline_status || 'DATE NOT AVAILABLE'} · ${formatDate(record.deadline)}`,
        record.status,
        record.application_url ? 'Official portal available' : 'Not available',
        record.documents_status === 'verified' ? (record.documents_display || []).join(', ') : 'Not specified',
      ].forEach(value => row.appendChild(textElement('td', '', value)));
      const sourceCell = document.createElement('td');
      sourceCell.appendChild(createOfficialLink('Official source', record.official_source_url));
      row.appendChild(sourceCell);
      compareTableBody.appendChild(row);
    });
    compareSection.classList.toggle('hidden', selected.length < 2);
  }

  function renderSavedPlanner() {
    savedList.replaceChildren();
    const savedRecords = plannerState.saved
      .map(id => catalogRecordById.get(id))
      .filter(Boolean);
    savedEmpty.classList.toggle('hidden', savedRecords.length > 0);
    savedRecords.forEach(record => {
      const panel = document.createElement('article');
      panel.className = 'saved-scholarship-card';
      panel.appendChild(textElement('h3', 'catalog-card-title', record.name));
      panel.appendChild(textElement('p', 'catalog-deadline', `${record.deadline_status || 'DATE NOT AVAILABLE'} · ${formatDate(record.deadline)}`));
      const entry = plannerState.entries[record.id] || { status: 'Not Started', steps: {} };
      const statusLabel = document.createElement('label');
      statusLabel.className = 'planner-status-label';
      statusLabel.appendChild(document.createTextNode('My status (manual): '));
      const statusSelect = document.createElement('select');
      statusSelect.className = 'form-select planner-status';
      APPLICATION_STATUSES.forEach(status => {
        const option = document.createElement('option');
        option.value = status;
        option.textContent = status;
        option.selected = (entry.status || 'Not Started') === status;
        statusSelect.appendChild(option);
      });
      statusSelect.addEventListener('change', () => {
        plannerState.entries[record.id] = { ...entry, status: statusSelect.value };
        savePlannerState();
      });
      statusLabel.appendChild(statusSelect);
      panel.appendChild(statusLabel);

      const checklist = document.createElement('ul');
      checklist.className = 'planner-checklist';
      PLANNER_STEPS.forEach((step, index) => {
        const item = document.createElement('li');
        const label = document.createElement('label');
        const checkbox = document.createElement('input');
        checkbox.type = 'checkbox';
        checkbox.checked = Boolean(entry.steps?.[index]);
        checkbox.addEventListener('change', () => {
          const current = plannerState.entries[record.id] || { status: 'Not Started', steps: {} };
          current.steps = { ...current.steps, [index]: checkbox.checked };
          plannerState.entries[record.id] = current;
          savePlannerState();
        });
        label.append(checkbox, document.createTextNode(step));
        item.appendChild(label);
        checklist.appendChild(item);
      });
      panel.appendChild(checklist);
      panel.appendChild(createOfficialLink('View official details', record.official_source_url));
      savedList.appendChild(panel);
    });
    updateCompareTable();
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
      Course: 'Engineering',
      ApplicationType: 'Not sure'
    },
    ineligible: {
      Gender: 'Male',
      Community: 'OC',
      FamilyIncome: 650000,
      '12thMarks': 55.0,
      FirstGraduate: 'No',
      District: 'Chennai',
      CollegeType: 'Private',
      Course: 'Management',
      ApplicationType: 'Not sure'
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
    document.getElementById('applicationType').value = profile.ApplicationType;

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
      Course: course,
      ApplicationType: document.getElementById('applicationType').value
    };

    // UI Loading State
    submitBtn.disabled = true;
    btnSpinner.classList.remove('hidden');
    document.querySelector('.btn-text').textContent = 'Evaluating Random Forest...';

    try {
      let response = await fetch('/api/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json'
        },
        cache: 'no-store',
        body: JSON.stringify(payload)
      });

      // If 404 on /api/predict, fallback to /api or /predict
      if (response.status === 404) {
        response = await fetch('/api', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
          },
          cache: 'no-store',
          body: JSON.stringify(payload)
        });
      }

      const contentType = response.headers.get('content-type') || '';
      let data;
      if (contentType.includes('application/json')) {
        data = await response.json();
      } else {
        const text = await response.text();
        console.error('Non-JSON response received:', text);
        throw new Error('API server returned a non-JSON response. Please check server status.');
      }

      if (!response.ok || (data.status && data.status === 'error')) {
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
    if (!data.input_summary || !Array.isArray(data.potential_scholarships)) {
      throw new Error('The prediction API returned an incomplete scholarship assessment.');
    }

    const isEligible = data.is_eligible !== undefined ? data.is_eligible : (data.prediction === 'Eligible');

    // Card Theme & Badges
    resultContainer.className = 'result-card ' + (isEligible ? 'eligible' : 'ineligible');
    resultBadge.className = 'result-badge ' + (isEligible ? 'eligible' : 'ineligible');

    if (isEligible) {
      resultIcon.innerHTML = '&#10003;'; // Checkmark
      resultTitle.textContent = 'ML PREDICTION: ELIGIBLE';
    } else {
      resultIcon.innerHTML = '&#10005;'; // Cross mark
      resultTitle.textContent = 'ML PREDICTION: NOT ELIGIBLE';
    }

    // Probability & Confidence Metrics (Supports both 0.94 and 94.0 format)
    const toPercentage = (value, label) => {
      const numericValue = Number(value);
      if (!Number.isFinite(numericValue)) {
        throw new Error(`The prediction API did not return a valid ${label}.`);
      }
      return numericValue <= 1 ? (numericValue * 100).toFixed(2) : numericValue;
    };
    const conf = toPercentage(data.confidence, 'confidence');
    const elProb = toPercentage(data.eligible_probability, 'eligible probability');
    const notElProb = toPercentage(data.not_eligible_probability, 'not-eligible probability');

    confidenceValue.textContent = `${conf}%`;
    eligibleProbValue.textContent = `${elProb}%`;
    notEligibleProbValue.textContent = `${notElProb}%`;

    // Dynamic Progress Bar Animations - reset first so re-evaluations animate distinctly
    eligibleProgress.style.width = '0%';
    notEligibleProgress.style.width = '0%';
    setTimeout(() => {
      eligibleProgress.style.width = `${elProb}%`;
      notEligibleProgress.style.width = `${notElProb}%`;
    }, 60);

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
      li.textContent = 'No model feature-attribution details were returned.';
      factorsList.appendChild(li);
    }

    estimatedSupportValue.textContent = data.estimated_support || 'No verified amount range is configured.';
    recommendationsList.replaceChildren();
    data.potential_scholarships.forEach(recommendation => {
      const item = document.createElement('li');
      item.className = 'recommendation-item';

      const heading = document.createElement('div');
      heading.className = 'recommendation-heading';
      const name = document.createElement('strong');
      name.textContent = recommendation.name;
      const type = document.createElement('span');
      type.className = 'recommendation-tag';
      type.textContent = recommendation.type || 'Possible Scholarship Category';
      heading.append(name, type);

      const reason = document.createElement('p');
      reason.className = 'recommendation-reason';
      reason.textContent = recommendation.reason;

      const amount = document.createElement('p');
      amount.className = 'recommendation-amount';
      amount.textContent = recommendation.estimated_amount
        ? `Estimated amount: ${recommendation.estimated_amount}`
        : 'Estimated amount: no verified range configured';

      item.append(heading, reason, amount);
      recommendationsList.appendChild(item);
    });
    renderMatches(data.scholarship_matches || []);
    if (data.scholarship_catalog_status === 'unavailable') {
      catalogWarning.textContent = 'The scholarship source catalog is unavailable. The ML prediction above was produced independently.';
      catalogWarning.classList.remove('hidden');
    } else {
      const unresolved = (data.scholarship_matches || []).some(record => record.status === 'Needs Verification');
      catalogWarning.textContent = unresolved
        ? 'Profile submitted: 8/8 model fields. Some official scholarship eligibility rules are not documented in this catalog, so individual eligibility and match scores remain unverified.'
        : '';
      catalogWarning.classList.toggle('hidden', !unresolved);
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

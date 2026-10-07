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
  const compareSort = document.getElementById('compare-sort');
  const stateFilter = document.getElementById('state-filter');
  const categoryFilter = document.getElementById('category-filter');
  const courseFilter = document.getElementById('course-filter');
  const communityFilter = document.getElementById('community-filter');
  const incomeFilter = document.getElementById('income-filter');
  const meritFilter = document.getElementById('merit-filter');
  const closingSoonFilter = document.getElementById('closing-soon-filter');
  const profileCompleteness = document.getElementById('profile-completeness');
  const whatIfSection = document.getElementById('what-if-section');
  const scenarioResult = document.getElementById('scenario-result');
  const scenarioError = document.getElementById('scenario-error');
  const instituteResults = document.getElementById('institute-results');
  const instituteCardsList = document.getElementById('institute-cards-list');

  const PLANNER_STEPS = [
    'Review the official eligibility guidelines',
    'Confirm the current application window',
    'Check the official document requirements',
    'Open the official application portal'
  ];
  const APPLICATION_STATUSES = [
    'Not Applied', 'Documents Collecting', 'Ready to Apply',
    'Applied', 'Institute Verification', 'Department Verification', 'Approved', 'Rejected'
  ];
  const PAYMENT_STATUSES = ['Not Available', 'Pending', 'Sanctioned', 'Disbursed'];
  const STORAGE_KEY = 'scholarshipAssistantPlannerV1';
  let plannerState = loadPlannerState();
  let catalogRecords = [];
  const catalogRecordById = new Map();
  let searchTimer;
  let catalogSearchSequence = 0;
  let currentAssessment = null;
  let currentProfile = null;

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
    if (val === null || val === undefined || val === '' || isNaN(val)) return '';
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
        entries: migratePlannerEntries(parsed.entries)
      };
    } catch (err) {
      console.error('Could not read locally saved scholarship planner data:', err);
      return { saved: [], compare: [], entries: {} };
    }

    function migratePlannerEntries(entries) {
      if (!entries || typeof entries !== 'object' || Array.isArray(entries)) return {};
      const migrated = {};
      Object.entries(entries).forEach(([id, entry]) => {
        if (!entry || typeof entry !== 'object') return;
        const oldStatuses = {
          'Not Started': 'Not Applied',
          'Verification Pending': 'Institute Verification'
        };
        const status = oldStatuses[entry.status] || entry.status;
        migrated[id] = {
          ...entry,
          status: APPLICATION_STATUSES.includes(status) ? status : 'Not Applied',
          paymentStatus: PAYMENT_STATUSES.includes(entry.paymentStatus) ? entry.paymentStatus : 'Not Available',
          steps: entry.steps && typeof entry.steps === 'object' ? entry.steps : {},
          documents: entry.documents && typeof entry.documents === 'object' ? entry.documents : {}
        };
      });
      return migrated;
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
    if (!href) return textElement('span', 'unknown-value', `${label}: not available in source`);
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
      plannerState.entries[id] = { status: 'Not Applied', paymentStatus: 'Not Available', steps: {}, documents: {} };
    }
    savePlannerState();
    renderCatalogCards(catalogRecords);
    renderSavedPlanner();
  }

  function toggleCompare(id, checked) {
    if (checked && !plannerState.compare.includes(id)) {
      if (plannerState.compare.length >= 4) {
        catalogWarning.textContent = 'Compare up to four scholarships at a time.';
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
    card.appendChild(textElement('p', 'catalog-provider', `${record.provider} · ${record.category}`));

    const badges = document.createElement('div');
    badges.className = 'catalog-badges';
    const deadlineStatus = record.deadline_status || '';
    const deadlineStatusLabel = {
      OPEN: 'Open',
      'CLOSING SOON': 'Closing Soon',
      CLOSED: 'Closed'
    }[deadlineStatus] || 'Date not available';
    badges.append(
      textElement('span', 'catalog-badge source-badge', record.source_status === 'official'
        ? 'Official source · verified record'
        : record.source_status === 'partial'
          ? 'Official source · partially verified'
          : 'Information requires verification'),
      textElement('span', `catalog-badge deadline-status status-${deadlineStatus.toLowerCase().replace(/\s+/g, '-') || 'unavailable'}`, deadlineStatusLabel)
    );
    card.appendChild(badges);

    card.appendChild(textElement('p', 'catalog-deadline', deadlineCardDescription(record)));
    const checklist = record.eligibility_checklist || [];
    const counts = checklist.reduce((total, item) => {
      total[item.status] = (total[item.status] || 0) + 1;
      return total;
    }, {});
    const matchSummary = document.createElement('div');
    matchSummary.className = 'catalog-match-summary';
    matchSummary.appendChild(textElement(
      'h4',
      'catalog-score',
      record.match_score === null || record.match_score === undefined
        ? 'Match: Not enough data'
        : `Match: ${record.match_score}/100`
    ));
    matchSummary.appendChild(textElement(
      'p',
      'catalog-match-explanation',
      record.match_score === null || record.match_score === undefined
        ? 'Some eligibility criteria are not documented, so a reliable match score cannot be calculated.'
        : 'Compatibility with documented eligibility criteria.'
    ));
    const checklistSummary = document.createElement('div');
    checklistSummary.className = 'catalog-checklist-summary';
    ['PASS', 'FAIL', 'UNKNOWN'].forEach(status => {
      checklistSummary.appendChild(textElement(
        'span',
        `checklist-count checklist-${status.toLowerCase()}`,
        `${status}: ${counts[status] || 0}`
      ));
    });
    matchSummary.appendChild(checklistSummary);
    const unknownCriteria = checklist
      .filter(item => item.status === 'UNKNOWN')
      .map(item => item.label);
    if (unknownCriteria.length) {
      matchSummary.appendChild(textElement(
        'p',
        'catalog-unknown-criteria',
        `Unknown criteria: ${unknownCriteria.join(', ')}`
      ));
    }
    card.appendChild(matchSummary);
    card.appendChild(textElement('p', 'catalog-verified-date', `Last verified: ${formatDate(record.last_verified)}`));

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
    if (record.official_source_url) {
      actions.appendChild(createOfficialLink('View Official Source', record.official_source_url));
    }
    card.appendChild(actions);

    const details = document.createElement('details');
    details.className = 'scholarship-details';
    details.appendChild(textElement('summary', '', 'Eligibility, benefits, documents and sources'));
    const detailGrid = document.createElement('div');
    detailGrid.className = 'scholarship-detail-grid';
    detailGrid.appendChild(textElement(
      'h4',
      'detail-subheading',
      record.failed_rules?.length ? 'WHY NOT A MATCH?' : 'WHY THIS SCHOLARSHIP?'
    ));
    const whyList = document.createElement('ul');
    whyList.className = 'why-checklist';
    (record.eligibility_checklist || []).forEach(criterion => {
      const symbol = criterion.status === 'PASS' ? '✓' : criterion.status === 'FAIL' ? '✗' : '⚠';
      const item = textElement(
        'li',
        `checklist-${criterion.status.toLowerCase()}`,
        `${symbol} ${criterion.status} · ${criterion.label}: ${criterion.reason}`
      );
      whyList.appendChild(item);
    });
    if (!whyList.childElementCount) {
      whyList.appendChild(textElement('li', 'checklist-unknown', '⚠ UNKNOWN · No documented eligibility rules are available.'));
    }
    detailGrid.appendChild(whyList);
    detailGrid.appendChild(textElement('p', 'eligibility-summary', record.eligibility_assessment || 'Cannot fully determine eligibility.'));
    if (record.missing_information?.length) {
      const missingTitle = textElement('h4', 'detail-subheading', 'Missing information');
      detailGrid.appendChild(missingTitle);
      const missingList = document.createElement('ul');
      record.missing_information.forEach(field => missingList.appendChild(textElement('li', '', field)));
      detailGrid.appendChild(missingList);
      const completeButton = textElement('button', 'btn btn-sm btn-outline', 'Complete Profile');
      completeButton.type = 'button';
      completeButton.addEventListener('click', () => {
        const targets = {
          Domicile: 'domicile',
          YearOfStudy: 'yearOfStudy',
          ApplicationType: 'applicationType',
          Gender: 'gender',
          Community: 'community',
          FamilyIncome: 'familyIncome',
          '12thMarks': 'twelfthMarks',
          FirstGraduate: 'firstGraduate',
          District: 'district',
          CollegeType: 'collegeType',
          Course: 'course'
        };
        const target = document.getElementById(targets[record.missing_information[0]] || 'gender');
        document.getElementById('predictor').scrollIntoView({ behavior: 'smooth' });
        target.focus({ preventScroll: true });
      });
      detailGrid.appendChild(completeButton);
    }
    detailGrid.append(
      textElement('div', 'detail-value', `Why this appears: ${(record.reasons || []).join(' ')}`),
      textElement('div', 'detail-value', `Failed requirements: ${(record.failed_rules || []).join('; ') || 'None documented.'}`),
      textElement('div', 'detail-value', `Unknown requirements: ${(record.unverified_criteria || []).join(', ') || 'None listed.'}`),
      textElement('div', 'detail-value', `Missing profile information: ${(record.missing_information || []).join(', ') || 'No profile fields are currently required by the partial source record.'}`),
      textElement('div', 'detail-value', `Benefit: ${record.benefits?.display || 'Not specified in the verified source.'}`),
      textElement('div', 'detail-value', `Application window: ${record.application_window || 'Not specified'}`),
      textElement('div', 'detail-value', `Required documents: ${record.documents_status === 'verified' ? (record.documents_display || []).join(', ') : 'Not specified in the verified source.'}`),
      textElement('div', 'detail-value', `Information last checked: ${formatDate(record.last_verified)}`)
    );
    detailGrid.appendChild(buildVerificationTimeline(record));
    detailGrid.appendChild(buildFaq(record));

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
    if (record.application_url) links.appendChild(createOfficialLink('Apply Officially · portal entry point', record.application_url));
    if (record.payment_tracking_url) {
      links.appendChild(createOfficialLink('Official Payment Tracking', record.payment_tracking_url));
    } else {
      links.appendChild(textElement('span', 'unknown-value', 'Official payment-tracking link: not available in source'));
    }
    detailGrid.appendChild(links);
    detailGrid.appendChild(textElement('p', 'detail-disclaimer', record.disclaimer || 'Always verify current official guidelines before applying.'));
    details.appendChild(detailGrid);
    card.appendChild(details);
    return card;
  }

  function deadlineCardDescription(record) {
    if (record.days_until_deadline === null || record.days_until_deadline === undefined || !record.deadline) {
      return 'Application deadline: Date not available in the verified source.';
    }
    const days = record.days_until_deadline;
    if (days < 0) return `Application deadline passed ${Math.abs(days)} day${days === -1 ? '' : 's'} ago · ${formatDate(record.deadline)}`;
    if (days === 0) return `Application closes today · ${formatDate(record.deadline)}`;
    return `Application closes in ${days} day${days === 1 ? '' : 's'} · ${formatDate(record.deadline)}`;
  }

  function deadlineDescription(record) {
    if (record.days_until_deadline === null || record.days_until_deadline === undefined) {
      return 'DATE NOT AVAILABLE · Student application deadline: Date not available';
    }
    const days = record.days_until_deadline;
    if (days < 0) return `CLOSED · Application deadline passed ${Math.abs(days)} day${Math.abs(days) === 1 ? '' : 's'} ago · ${formatDate(record.deadline)}`;
    if (days === 0) return `CLOSING SOON · Application closes today · ${formatDate(record.deadline)}`;
    if (days <= 7) return `CLOSING SOON · Application closes in ${days} day${days === 1 ? '' : 's'} · ${formatDate(record.deadline)}`;
    return `OPEN · Application closes in ${days} days · ${formatDate(record.deadline)}`;
  }

  function buildVerificationTimeline(record) {
    const section = document.createElement('section');
    section.className = 'verification-timeline';
    section.appendChild(textElement('h4', 'detail-subheading', 'Verification Timeline'));
    const items = [
      ['Student Application Deadline', record.deadline],
      ['Defective Application Verification', record.verification_deadlines?.defective_application],
      ['Institute Verification', record.verification_deadlines?.institution],
      ['Department Verification', record.verification_deadlines?.department],
      ['Level 2 Verification', record.verification_deadlines?.level_2],
      ['Final Processing / Disbursement', record.verification_deadlines?.final_processing]
    ];
    const list = document.createElement('ol');
    items.forEach(([label, value]) => {
      const item = textElement('li', '', `${label}: ${formatDate(value)}`);
      list.appendChild(item);
    });
    section.appendChild(list);
    return section;
  }

  function buildFaq(record) {
    const faq = document.createElement('details');
    faq.className = 'scholarship-faq';
    faq.appendChild(textElement('summary', '', 'Scholarship FAQ · answers from stored source data'));
    const items = [
      ['Who can apply?', record.eligibility_rules_status === 'verified' && record.eligibility_checklist?.length
        ? record.eligibility_checklist.map(rule => `${rule.label}: ${rule.status}`).join('; ')
        : 'Not specified in the available source.'],
      ['What is the income limit?', record.eligibility_criteria?.income_limit === null
        ? 'Not specified in the available source.'
        : `₹${Number(record.eligibility_criteria.income_limit).toLocaleString('en-IN')} annual family income.`],
      ['What benefits are provided?', record.benefits?.display || 'Not specified in the available source.'],
      ['What documents are required?', record.documents_status === 'verified'
        ? (record.documents_display || []).join(', ')
        : 'Not specified in the available source.'],
      ['What is the deadline?', record.deadline ? `${formatDate(record.deadline)} (${record.deadline_status}).` : 'Not specified in the available source.'],
      ['Where do I apply?', record.application_url ? 'The linked portal is an official general entry point; scheme-specific application availability is not confirmed.' : 'Not specified in the available source.'],
      ['How is the application verified?', record.verification_deadlines?.institution || record.verification_deadlines?.level_2
        ? `Institute: ${formatDate(record.verification_deadlines?.institution)}; Level 2: ${formatDate(record.verification_deadlines?.level_2)}.`
        : 'Not specified in the available source.']
    ];
    const list = document.createElement('dl');
    items.forEach(([question, answer]) => {
      list.append(textElement('dt', '', question), textElement('dd', '', answer));
    });
    faq.appendChild(list);
    return faq;
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
    if (stateFilter.value) params.set('state', stateFilter.value);
    if (categoryFilter.value) params.set('category', categoryFilter.value);
    if (courseFilter.value) params.set('course', courseFilter.value);
    if (communityFilter.value) params.set('community', communityFilter.value);
    if (incomeFilter.checked) params.set('income_based', 'true');
    if (meritFilter.checked) params.set('merit_based', 'true');
    if (openOnlyFilter.checked) params.set('open', 'true');
    if (closingSoonFilter.checked) params.set('closing_soon', 'true');
    const searchId = ++catalogSearchSequence;
    try {
      const response = await fetch(`/api/scholarships?${params.toString()}`, { cache: 'no-store' });
      const result = await response.json();
      if (searchId !== catalogSearchSequence) return;
      if (!response.ok || result.status !== 'success') {
        throw new Error(result.message || 'Scholarship catalog could not be loaded.');
      }
      catalogRecords = result.scholarships;
      if (currentProfile) {
        const recommendationResponse = await fetch('/api/scholarships/recommend', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json', 'Accept': 'application/json' },
          cache: 'no-store',
          body: JSON.stringify({
            profile: currentProfile,
            filters: Object.fromEntries(params.entries())
          })
        });
        const personalized = await recommendationResponse.json();
        if (!recommendationResponse.ok || personalized.status !== 'success') {
          throw new Error(personalized.message || 'Profile-based catalog filtering failed.');
        }
        catalogRecords = personalized.scholarship_matches;
      }
      if (searchId !== catalogSearchSequence) return;
      catalogRecords.forEach(record => catalogRecordById.set(record.id, record));
      if (result.filter_options) {
        setFilterOptions(stateFilter, result.filter_options.states, 'Any supported level');
        setFilterOptions(categoryFilter, result.filter_options.categories, 'Any documented category');
        setFilterOptions(courseFilter, result.filter_options.courses, 'Any documented course');
        setFilterOptions(communityFilter, result.filter_options.communities, 'Any documented community');
      }
      catalogEmpty.textContent = 'No catalog entries match these search filters.';
      renderCatalogCards(catalogRecords);
      catalogUpdated.textContent = `Reviewed catalog updated ${formatDate(result.database_last_updated)} · ${result.count} records`;
      catalogWarning.classList.add('hidden');
      renderSavedPlanner();
      loadCatalogQuality();
      renderHelpSource();
    } catch (err) {
      if (searchId !== catalogSearchSequence) return;
      console.error('Scholarship catalog request failed:', err);
      catalogGrid.replaceChildren();
      catalogEmpty.textContent = 'Scholarship source data is temporarily unavailable. Eligibility screening remains available.';
      catalogEmpty.classList.remove('hidden');
      catalogUpdated.textContent = 'Source verification unavailable.';
    }

    function setFilterOptions(select, values, placeholder) {
      const previous = select.value;
      select.replaceChildren(new Option(placeholder, ''));
      (values || []).forEach(value => select.add(new Option(value, value)));
      select.value = (values || []).includes(previous) ? previous : '';
    }

    async function loadCatalogQuality() {
      try {
        const response = await fetch('/api/catalog/quality', { cache: 'no-store' });
        const report = await response.json();
        if (response.ok && report.status === 'success' && report.warning_count > 0) {
          catalogUpdated.textContent += ` · ${report.warning_count} source/data-quality warning(s)`;
          const warningList = document.getElementById('catalog-quality-warnings');
          warningList.replaceChildren();
          report.warnings.forEach(warning => warningList.appendChild(textElement('li', '', warning)));
          document.getElementById('catalog-quality-details').classList.remove('hidden');
        }
      } catch (error) {
        console.warn('Catalog quality report unavailable:', error);
      }
    }

    function renderHelpSource() {
      const container = document.getElementById('help-source-links');
      container.replaceChildren();
      const source = [...catalogRecordById.values()].find(record => record.source?.url)?.source;
      if (source) {
        container.appendChild(createOfficialLink(
          `${source.name} · general official portal`,
          source.url
        ));
        container.appendChild(textElement('p', 'detail-value', source.limitations));
      } else {
        container.appendChild(textElement('p', 'unknown-value', 'No verified help or grievance links are present in the source catalog.'));
      }
    }
  }

  catalogSearch.addEventListener('input', () => {
    clearTimeout(searchTimer);
    searchTimer = setTimeout(loadScholarships, 250);
  });
  [
    openOnlyFilter, closingSoonFilter, incomeFilter, meritFilter,
    stateFilter, categoryFilter, courseFilter, communityFilter
  ].forEach(control => control.addEventListener('change', loadScholarships));
  [stateFilter, categoryFilter, courseFilter, communityFilter].forEach(control => {
    control.addEventListener('focus', () => {
      if (control.options.length <= 1) {
        catalogWarning.textContent = 'No values for this filter are documented in the current source snapshot.';
        catalogWarning.classList.remove('hidden');
      }
    });
  });

  document.getElementById('institute-search-button').addEventListener('click', async () => {
    const params = new URLSearchParams({
      name: document.getElementById('institute-search-name').value.trim(),
      district: document.getElementById('institute-search-district').value.trim(),
      state: document.getElementById('institute-search-state').value.trim(),
      type: document.getElementById('institute-search-type').value.trim(),
      course: document.getElementById('institute-search-course').value.trim()
    });
    const searchBtn = document.getElementById('institute-search-button');
    searchBtn.disabled = true;
    try {
      const response = await fetch(`/api/institutes?${params.toString()}`, { cache: 'no-store' });
      const result = await response.json();
      if (!response.ok) throw new Error(result.message || 'Institute search failed.');
      if (Array.isArray(result.institutes) && result.institutes.length > 0) {
        instituteResults.textContent = `${result.count || result.institutes.length} source-linked institute records found.`;
        if (instituteCardsList) {
          instituteCardsList.classList.remove('hidden');
          instituteCardsList.replaceChildren();
          result.institutes.forEach(inst => {
            const card = document.createElement('article');
            card.className = 'catalog-card';
            const title = document.createElement('h3');
            title.className = 'catalog-title';
            title.textContent = inst.name || 'Unnamed Institution';
            card.appendChild(title);

            const meta = document.createElement('div');
            meta.className = 'catalog-badges';
            if (inst.type) {
              const typeBadge = document.createElement('span');
              typeBadge.className = 'catalog-badge category-badge';
              typeBadge.textContent = inst.type;
              meta.appendChild(typeBadge);
            }
            if (inst.district || inst.state) {
              const locBadge = document.createElement('span');
              locBadge.className = 'catalog-badge state-badge';
              locBadge.textContent = [inst.district, inst.state].filter(Boolean).join(', ');
              meta.appendChild(locBadge);
            }
            card.appendChild(meta);

            if (inst.courses || inst.course) {
              const coursesText = Array.isArray(inst.courses) ? inst.courses.join(', ') : (inst.course || '');
              const p = document.createElement('p');
              p.className = 'catalog-provider';
              p.textContent = `Courses: ${coursesText}`;
              card.appendChild(p);
            }
            instituteCardsList.appendChild(card);
          });
        }
      } else {
        instituteResults.textContent = result.count
          ? `${result.count} source-linked institute records found.`
          : (result.message || 'No authoritative institute directory is configured. No institute records are shown.');
        if (instituteCardsList) {
          instituteCardsList.classList.add('hidden');
          instituteCardsList.replaceChildren();
        }
      }
    } catch (error) {
      console.error('Institute directory search failed:', error);
      instituteResults.textContent = 'Institute search is unavailable. Verify institution information using an official source.';
      if (instituteCardsList) {
        instituteCardsList.classList.add('hidden');
      }
    } finally {
      searchBtn.disabled = false;
    }
  });

  const instituteSearchNameInput = document.getElementById('institute-search-name');
  if (instituteSearchNameInput) {
    instituteSearchNameInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') {
        e.preventDefault();
        document.getElementById('institute-search-button').click();
      }
    });
  }

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
    const matching = new Map(records.map(record => [record.id, record]));
    catalogRecords = catalogRecords.map(record => matching.get(record.id) || record);
    renderCatalogCards(catalogRecords);
    renderSavedPlanner();
  }

  function updateCompareTable() {
    const selected = plannerState.compare
      .map(id => catalogRecordById.get(id))
      .filter(Boolean);
    const sortMode = compareSort.value;
    selected.sort((left, right) => {
      if (sortMode === 'score') {
        return (right.match_score ?? -1) - (left.match_score ?? -1);
      }
      if (sortMode === 'deadline') {
        return (left.days_until_deadline ?? Number.POSITIVE_INFINITY)
          - (right.days_until_deadline ?? Number.POSITIVE_INFINITY);
      }
      const leftKnown = left.benefits?.amount !== null || Boolean(left.benefits?.description);
      const rightKnown = right.benefits?.amount !== null || Boolean(right.benefits?.description);
      const numericBenefit = value => {
        if (typeof value === 'number' && Number.isFinite(value)) return value;
        if (typeof value !== 'string') return null;
        const amounts = value.match(/\d[\d,]*(?:\.\d+)?/g);
        if (!amounts?.length) return null;
        const parsed = amounts.map(amount => Number(amount.replace(/,/g, '')));
        return parsed.every(Number.isFinite)
          ? parsed.reduce((sum, amount) => sum + amount, 0) / parsed.length
          : null;
      };
      const leftAmount = numericBenefit(left.benefits?.amount);
      const rightAmount = numericBenefit(right.benefits?.amount);
      if (leftAmount !== null && rightAmount !== null && leftAmount !== rightAmount) {
        return rightAmount - leftAmount;
      }
      return Number(rightKnown) - Number(leftKnown);
    });
    compareTableBody.replaceChildren();
    const header = document.getElementById('compare-table-header');
    header.replaceChildren(textElement('th', '', 'Feature'));
    selected.forEach(record => header.appendChild(textElement('th', '', record.name)));
    const features = [
      ['Match score', record => record.match_score === null ? 'Not available' : `${record.match_score}/100`],
      ['Eligibility assessment', record => record.eligibility_assessment],
      ['Benefit', record => record.benefits?.display || 'Not specified'],
      ['Income limit', record => formatRuleValue(record.eligibility_criteria?.income_limit, 'income')],
      ['Eligible courses', record => formatRuleValue(record.eligibility_criteria?.eligible_courses)],
      ['Eligible communities', record => formatRuleValue(record.eligibility_criteria?.eligible_communities)],
      ['Institution type', record => formatRuleValue(record.eligibility_criteria?.eligible_college_types)],
      ['Deadline', record => deadlineDescription(record)],
      ['Required documents', record => record.documents_status === 'verified'
        ? (record.documents_display || []).join(', ') || 'No documents listed'
        : 'Not specified in source'],
      ['Application portal', record => record.application_url ? 'Official general portal linked' : 'Not available'],
      ['Application status', record => plannerState.entries[record.id]?.status || 'Not Applied']
    ];
    features.forEach(([label, valueFor]) => {
      const row = document.createElement('tr');
      row.appendChild(textElement('th', '', label));
      selected.forEach(record => row.appendChild(textElement('td', '', valueFor(record) || 'Not specified')));
      compareTableBody.appendChild(row);
    });
    compareSection.classList.toggle('hidden', selected.length < 2);
  }

  function formatRuleValue(value, kind = 'list') {
    if (value === null || value === undefined) return 'Not specified in source';
    if (kind === 'income') return `₹${Number(value).toLocaleString('en-IN')} / year`;
    return Array.isArray(value) ? value.join(', ') : String(value);
  }

  compareSort.addEventListener('change', updateCompareTable);

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
      panel.appendChild(textElement('p', 'catalog-score', record.match_score === null
        ? 'Match score unavailable — eligibility rules are incomplete'
        : `${record.match_score}/100 documented-criteria compatibility`));
      panel.appendChild(textElement('p', 'detail-value', `Benefit: ${record.benefits?.display || 'Not specified in the verified source.'}`));
      panel.appendChild(textElement('p', 'catalog-deadline', deadlineDescription(record)));
      const entry = plannerState.entries[record.id] || {
        status: 'Not Applied', paymentStatus: 'Not Available', steps: {}, documents: {}
      };
      plannerState.entries[record.id] = entry;
      const statusLabel = document.createElement('label');
      statusLabel.className = 'planner-status-label';
      statusLabel.appendChild(document.createTextNode('My status (manual): '));
      const statusSelect = document.createElement('select');
      statusSelect.className = 'form-select planner-status';
      APPLICATION_STATUSES.forEach(status => {
        const option = document.createElement('option');
        option.value = status;
        option.textContent = status;
        option.selected = (entry.status || 'Not Applied') === status;
        statusSelect.appendChild(option);
      });
      statusSelect.addEventListener('change', () => {
        plannerState.entries[record.id] = { ...plannerState.entries[record.id], status: statusSelect.value };
        savePlannerState();
        updateCompareTable();
      });
      statusLabel.appendChild(statusSelect);
      panel.appendChild(statusLabel);

      const paymentLabel = document.createElement('label');
      paymentLabel.className = 'planner-status-label';
      paymentLabel.appendChild(document.createTextNode('Payment status (user-entered): '));
      const paymentSelect = document.createElement('select');
      paymentSelect.className = 'form-select planner-payment-status';
      PAYMENT_STATUSES.forEach(status => {
        const option = new Option(status, status, false, (entry.paymentStatus || 'Not Available') === status);
        paymentSelect.add(option);
      });
      paymentSelect.addEventListener('change', () => {
        plannerState.entries[record.id] = {
          ...plannerState.entries[record.id],
          paymentStatus: paymentSelect.value
        };
        savePlannerState();
      });
      paymentLabel.appendChild(paymentSelect);
      panel.appendChild(paymentLabel);
      panel.appendChild(textElement(
        'p',
        'payment-disclaimer',
        'Payment status shown here is user-entered unless connected to an official API.'
      ));
      panel.appendChild(record.payment_tracking_url
        ? createOfficialLink('Official payment tracking', record.payment_tracking_url)
        : textElement('p', 'unknown-value', 'Official payment-tracking link: not available in source.'));

      const savedDetails = createScholarshipCard(record, 'saved').querySelector('.scholarship-details');
      if (savedDetails) {
        const detailsToggle = document.createElement('details');
        detailsToggle.className = 'saved-details';
        detailsToggle.appendChild(textElement('summary', '', 'Open scholarship details'));
        detailsToggle.appendChild(savedDetails);
        panel.appendChild(detailsToggle);
      }

      const checklist = document.createElement('ul');
      checklist.className = 'planner-checklist';
      panel.appendChild(textElement('h4', 'detail-subheading', 'Application planning steps'));
      PLANNER_STEPS.forEach((step, index) => {
        const item = document.createElement('li');
        const label = document.createElement('label');
        const checkbox = document.createElement('input');
        checkbox.type = 'checkbox';
        checkbox.checked = Boolean(entry.steps?.[index]);
        checkbox.addEventListener('change', () => {
          const current = plannerState.entries[record.id] || {
            status: 'Not Applied', paymentStatus: 'Not Available', steps: {}, documents: {}
          };
          current.steps = { ...current.steps, [index]: checkbox.checked };
          plannerState.entries[record.id] = current;
          savePlannerState();
        });
        label.append(checkbox, document.createTextNode(step));
        item.appendChild(label);
        checklist.appendChild(item);
      });
      panel.appendChild(checklist);
      panel.appendChild(textElement('h4', 'detail-subheading', 'Official document checklist'));
      if (record.documents_status === 'verified') {
        const documents = record.documents_display || [];
        const completedCount = documents.filter(name => Boolean(entry.documents?.[name])).length;
        panel.appendChild(textElement('p', 'document-progress', `${completedCount} / ${documents.length} documents ready`));
        const documentList = document.createElement('ul');
        documentList.className = 'planner-checklist';
        documents.forEach(name => {
          const label = document.createElement('label');
          const checkbox = document.createElement('input');
          checkbox.type = 'checkbox';
          checkbox.checked = Boolean(entry.documents?.[name]);
          checkbox.addEventListener('change', () => {
            const current = plannerState.entries[record.id];
            current.documents = { ...current.documents, [name]: checkbox.checked };
            savePlannerState();
            renderSavedPlanner();
          });
          label.append(checkbox, document.createTextNode(name));
          documentList.appendChild(textElement('li', '', ''));
          documentList.lastChild.appendChild(label);
        });
        panel.appendChild(documentList);
      } else {
        panel.appendChild(textElement(
          'p',
          'unknown-value',
          'Document requirements are not specified in the available official source. No sample document list is assumed.'
        ));
      }
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
    whatIfSection.classList.add('hidden');
    currentAssessment = null;
    currentProfile = null;
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
    const domicile = document.getElementById('domicile').value.trim();
    const yearOfStudy = document.getElementById('yearOfStudy').value.trim();
    if (domicile) payload.Domicile = domicile;
    if (yearOfStudy) payload.YearOfStudy = yearOfStudy;

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
    currentAssessment = data;
    currentProfile = { ...data.input_summary };

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
    const discoveryMissing = [...new Set((data.scholarship_matches || [])
      .flatMap(record => record.missing_information || []))];
    const modelFields = [
      'Gender', 'Community', 'FamilyIncome', '12thMarks',
      'FirstGraduate', 'District', 'CollegeType', 'Course'
    ];
    const providedCount = modelFields.filter(field => data.input_summary[field] !== undefined).length;
    profileCompleteness.replaceChildren(
      textElement('strong', '', `Profile completeness: ${Math.round(100 * providedCount / modelFields.length)}% (${providedCount}/${modelFields.length} Random Forest inputs)`),
      textElement('p', '', discoveryMissing.length
        ? `Additional information requested by documented rules: ${discoveryMissing.join(', ')}.`
        : 'No additional profile fields are required by the currently documented rules. Unverified scheme criteria remain unknown.')
    );
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
    summaryGrid.replaceChildren();
    [
      ['Gender', summary.Gender],
      ['Community', summary.Community],
      ['Family Income', formatINR(summary.FamilyIncome)],
      ['12th Marks', `${summary['12thMarks']}%`],
      ['First Graduate', summary.FirstGraduate],
      ['District', summary.District],
      ['College Type', summary.CollegeType],
      ['Course', summary.Course],
      ['Domicile (discovery only)', summary.Domicile],
      ['Year of Study (discovery only)', summary.YearOfStudy],
      ['Application Type (discovery only)', summary.ApplicationType]
    ].filter(([, value]) => value !== undefined && value !== '').forEach(([label, value]) => {
      const item = document.createElement('div');
      item.className = 'summary-item';
      item.append(textElement('span', 'summary-item-label', label));
      item.append(textElement('span', 'summary-item-value', value));
      summaryGrid.appendChild(item);
    });

    // Reveal and scroll smoothly
    document.getElementById('scenario-marks').value = summary['12thMarks'];
    document.getElementById('scenario-income').value = summary.FamilyIncome;
    document.getElementById('scenario-community').value = summary.Community;
    document.getElementById('scenario-first-graduate').value = summary.FirstGraduate;
    document.getElementById('scenario-course').value = summary.Course;
    document.getElementById('scenario-college-type').value = summary.CollegeType;
    scenarioResult.classList.add('hidden');
    scenarioError.classList.add('hidden');
    whatIfSection.classList.remove('hidden');
    resultContainer.classList.remove('hidden');
    resultContainer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }

  document.getElementById('recalculate-scenario').addEventListener('click', async () => {
    if (!currentProfile || !currentAssessment) return;
    scenarioError.classList.add('hidden');
    const scenarioProfile = {
      ...currentProfile,
      '12thMarks': Number(document.getElementById('scenario-marks').value),
      FamilyIncome: Number(document.getElementById('scenario-income').value),
      Community: document.getElementById('scenario-community').value,
      FirstGraduate: document.getElementById('scenario-first-graduate').value,
      Course: document.getElementById('scenario-course').value,
      CollegeType: document.getElementById('scenario-college-type').value
    };
    if (
      !Number.isFinite(scenarioProfile['12thMarks'])
      || scenarioProfile['12thMarks'] < 0 || scenarioProfile['12thMarks'] > 100
      || !Number.isFinite(scenarioProfile.FamilyIncome)
      || scenarioProfile.FamilyIncome < 0 || scenarioProfile.FamilyIncome > 10000000
    ) {
      scenarioError.textContent = 'Enter valid marks (0–100) and annual family income (₹0–₹1,00,00,000).';
      scenarioError.classList.remove('hidden');
      return;
    }
    const button = document.getElementById('recalculate-scenario');
    button.disabled = true;
    try {
      const response = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'Accept': 'application/json' },
        cache: 'no-store',
        body: JSON.stringify(scenarioProfile)
      });
      const result = await response.json();
      if (!response.ok || result.status !== 'success') {
        throw new Error(result.message || 'The scenario could not be recalculated.');
      }
      renderScenarioComparison(currentAssessment, result, scenarioProfile);
    } catch (error) {
      scenarioError.textContent = error.message || 'The scenario could not be recalculated.';
      scenarioError.classList.remove('hidden');
    } finally {
      button.disabled = false;
    }
  });

  function renderScenarioComparison(original, scenario, profile) {
    scenarioResult.replaceChildren();
    const columns = document.createElement('div');
    columns.className = 'scenario-columns';
    [
      ['Original Prediction', original],
      ['What-If Prediction', scenario]
    ].forEach(([label, assessment]) => {
      const card = document.createElement('section');
      card.className = 'scenario-card';
      card.append(
        textElement('h3', '', label),
        textElement('p', '', assessment.prediction),
        textElement('p', '', `Random Forest model probability: ${assessment.confidence}%`),
        textElement('p', '', `Eligible probability: ${assessment.eligible_probability}%`),
        textElement('p', '', `Marks: ${assessment.input_summary['12thMarks']}% · Income: ${formatINR(assessment.input_summary.FamilyIncome)}`),
        textElement('p', '', `Community: ${assessment.input_summary.Community} · Course: ${assessment.input_summary.Course}`),
        textElement('p', '', `Scholarship records: ${(assessment.scholarship_matches || []).map(match => `${match.name} — ${match.status}`).join('; ') || 'Unavailable'}`)
      );
      columns.appendChild(card);
    });
    scenarioResult.appendChild(columns);
    scenarioResult.appendChild(textElement(
      'p',
      'scenario-disclaimer',
      'Scenario inputs were sent to the existing model pipeline. Identical class labels are valid when probabilities differ; this is not an award prediction.'
    ));
    scenarioResult.classList.remove('hidden');
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

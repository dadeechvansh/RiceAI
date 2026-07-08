/* =========================================================================
   RiceAI — script.js
   Production frontend logic for the RiceAI FastAPI + Jinja2 application.

   Sections:
     1. Utilities
     2. Navbar (sticky, blur-on-scroll, mobile toggle, active link)
     3. Button ripple + hover-glow effects
     4. Scroll reveal animations (IntersectionObserver)
     5. Statistics counter animation (data-count)
     6. Timeline progress animation
     7. Predict page (sample fill, form submit, result states, reset)
   ========================================================================= */

document.addEventListener('DOMContentLoaded', () => {
  initNavbar();
  initButtonEffects();
  initRevealAnimations();
  initStatCounters();
  initTimelineProgress();
  initPredictPage();
});

/* =========================================================================
   1. UTILITIES
   ========================================================================= */

/**
 * Clamp a number between a min and max value.
 */
function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

/**
 * Ease-out cubic easing function, used for JS-driven number/width animations.
 */
function easeOutCubic(t) {
  return 1 - Math.pow(1 - t, 3);
}

/**
 * Run a numeric animation from `from` to `to` over `duration` ms,
 * calling `onUpdate` on every frame with the current value.
 * Returns a Promise that resolves when the animation completes.
 */
function animateValue({ from, to, duration = 1000, onUpdate, onComplete }) {
  return new Promise((resolve) => {
    const start = performance.now();

    function frame(now) {
      const elapsed = now - start;
      const progress = clamp(elapsed / duration, 0, 1);
      const eased = easeOutCubic(progress);
      const current = from + (to - from) * eased;

      onUpdate(current);

      if (progress < 1) {
        requestAnimationFrame(frame);
      } else {
        onUpdate(to);
        if (onComplete) onComplete();
        resolve();
      }
    }

    requestAnimationFrame(frame);
  });
}

/* =========================================================================
   2. NAVBAR
   ========================================================================= */

function initNavbar() {
  const header = document.getElementById('siteHeader');
  const navToggle = document.getElementById('navToggle');
  const navLinks = document.getElementById('navLinks');

  if (!header) return;

  /* ---- Sticky navbar blur-on-scroll ---- */
  const SCROLL_THRESHOLD = 12;

  function updateHeaderState() {
    if (window.scrollY > SCROLL_THRESHOLD) {
      header.classList.add('is-scrolled');
    } else {
      header.classList.remove('is-scrolled');
    }
  }

  updateHeaderState();
  window.addEventListener('scroll', updateHeaderState, { passive: true });

  /* ---- Mobile navigation toggle ---- */
  if (navToggle && navLinks) {
    navToggle.addEventListener('click', () => {
      const isOpen = navLinks.classList.toggle('is-open');
      navToggle.setAttribute('aria-expanded', String(isOpen));
    });

    /* Close mobile menu after clicking a navigation link */
    navLinks.querySelectorAll('a').forEach((link) => {
      link.addEventListener('click', () => {
        navLinks.classList.remove('is-open');
        navToggle.setAttribute('aria-expanded', 'false');
      });
    });

    /* Close mobile menu when clicking outside of it */
    document.addEventListener('click', (event) => {
      const isClickInsideNav = navLinks.contains(event.target) || navToggle.contains(event.target);
      if (!isClickInsideNav && navLinks.classList.contains('is-open')) {
        navLinks.classList.remove('is-open');
        navToggle.setAttribute('aria-expanded', 'false');
      }
    });
  }

  /* ---- Smooth scrolling for in-page anchor links ---- */
  document.querySelectorAll('a[href^="#"]').forEach((anchor) => {
    anchor.addEventListener('click', (event) => {
      const targetId = anchor.getAttribute('href');
      if (!targetId || targetId === '#') return;

      const targetEl = document.querySelector(targetId);
      if (targetEl) {
        event.preventDefault();
        targetEl.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }
    });
  });

  /* ---- Active navigation highlighting based on current path ---- */
  const currentPath = window.location.pathname.replace(/\/+$/, '') || '/';
  navLinks?.querySelectorAll('.nav-link').forEach((link) => {
    const linkPath = link.getAttribute('href');
    if (!linkPath || linkPath.startsWith('http')) return;

    const normalizedLinkPath = linkPath.replace(/\/+$/, '') || '/';
    if (normalizedLinkPath === currentPath) {
      link.classList.add('nav-link--active');
    }
  });
}

/* =========================================================================
   3. BUTTON EFFECTS (ripple + hover-glow)
   ========================================================================= */

function initButtonEffects() {
  const buttons = document.querySelectorAll('.btn');

  buttons.forEach((btn) => {
    btn.addEventListener('click', (event) => {
      triggerRipple(btn, event);
    });
  });
}

/**
 * Adds the `.rippling` class (defined in CSS) to a button briefly to
 * trigger the ripple pseudo-element animation, then removes it.
 */
function triggerRipple(button, event) {
  button.classList.remove('rippling');
  // Force reflow so the animation can be re-triggered on rapid clicks.
  void button.offsetWidth;
  button.classList.add('rippling');

  window.setTimeout(() => {
    button.classList.remove('rippling');
  }, 650);
}

/* =========================================================================
   4. SCROLL REVEAL ANIMATIONS
   ========================================================================= */

function initRevealAnimations() {
  const revealEls = document.querySelectorAll('.reveal');
  if (!revealEls.length) return;

  if (!('IntersectionObserver' in window)) {
    // Fallback: reveal everything immediately.
    revealEls.forEach((el) => el.classList.add('is-visible'));
    return;
  }

  const observer = new IntersectionObserver(
    (entries, obs) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add('is-visible');
          obs.unobserve(entry.target);
        }
      });
    },
    {
      threshold: 0.12,
      rootMargin: '0px 0px -60px 0px',
    }
  );

  revealEls.forEach((el) => observer.observe(el));
}

/* =========================================================================
   5. STATISTICS COUNTER ANIMATION
   ========================================================================= */

function initStatCounters() {
  const counters = document.querySelectorAll('[data-count]');
  if (!counters.length) return;

  if (!('IntersectionObserver' in window)) {
    counters.forEach((el) => setCounterFinalValue(el));
    return;
  }

  const observer = new IntersectionObserver(
    (entries, obs) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          animateCounter(entry.target);
          obs.unobserve(entry.target);
        }
      });
    },
    { threshold: 0.4 }
  );

  counters.forEach((el) => observer.observe(el));
}

/**
 * Animates a single [data-count] element from 0 to its target value.
 * Supports integers and decimals via the optional [data-decimals] attribute.
 */
function animateCounter(el) {
  const target = parseFloat(el.getAttribute('data-count'));
  if (Number.isNaN(target)) return;

  const decimals = parseInt(el.getAttribute('data-decimals'), 10) || 0;
  const duration = 1600;

  animateValue({
    from: 0,
    to: target,
    duration,
    onUpdate: (value) => {
      el.textContent = formatCounterValue(value, decimals);
    },
    onComplete: () => {
      el.textContent = formatCounterValue(target, decimals);
    },
  });
}

/**
 * Immediately sets a counter element to its final value (no animation).
 * Used as a fallback when IntersectionObserver is unavailable.
 */
function setCounterFinalValue(el) {
  const target = parseFloat(el.getAttribute('data-count'));
  if (Number.isNaN(target)) return;
  const decimals = parseInt(el.getAttribute('data-decimals'), 10) || 0;
  el.textContent = formatCounterValue(target, decimals);
}

/**
 * Formats a numeric value with thousands separators and fixed decimals.
 */
function formatCounterValue(value, decimals) {
  if (decimals > 0) {
    return value.toLocaleString('en-US', {
      minimumFractionDigits: decimals,
      maximumFractionDigits: decimals,
    });
  }
  return Math.round(value).toLocaleString('en-US');
}

/* =========================================================================
   6. TIMELINE PROGRESS ANIMATION
   ========================================================================= */

function initTimelineProgress() {
  const timeline = document.querySelector('.timeline');
  if (!timeline) return;

  if (!('IntersectionObserver' in window)) {
    timeline.classList.add('in-view');
    return;
  }

  const observer = new IntersectionObserver(
    (entries, obs) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          timeline.classList.add('in-view');
          obs.unobserve(entry.target);
        }
      });
    },
    { threshold: 0.25 }
  );

  observer.observe(timeline);
}

/* =========================================================================
   7. PREDICT PAGE
   ========================================================================= */

const SAMPLE_GRAIN = {
  Area: 15231,
  Perimeter: 525.578979,
  Major_Axis_Length: 229.749878,
  Minor_Axis_Length: 85.093788,
  Eccentricity: 0.928882,
  Convex_Area: 15617,
  Extent: 0.572896,
};

function initPredictPage() {
  const form = document.getElementById('predictForm');
  if (!form) return; // Not on the predict page — nothing to do.

  const fillSampleBtn = document.getElementById('fillSample');
  const submitBtn = document.getElementById('submitBtn');
  const resetBtn = document.getElementById('resetBtn');
  const retryBtn = document.getElementById('retryBtn');

  const stateIdle = document.getElementById('stateIdle');
  const stateLoading = document.getElementById('stateLoading');
  const stateSuccess = document.getElementById('stateSuccess');
  const stateError = document.getElementById('stateError');

  const predictionPill = document.getElementById('predictionPill');
  const predictionLabel = document.getElementById('predictionLabel');
  const confidenceValue = document.getElementById('confidenceValue');
  const confidenceFill = document.getElementById('confidenceFill');
  const errorMessage = document.getElementById('errorMessage');

  const resultStates = { stateIdle, stateLoading, stateSuccess, stateError };

  /* ---- Use sample grain button ---- */
  if (fillSampleBtn) {
    fillSampleBtn.addEventListener('click', () => {
      fillSampleValues(form);
    });
  }

  /* ---- Form submission ---- */
  form.addEventListener('submit', (event) => {
    event.preventDefault();
    handlePredictSubmit({
      form,
      submitBtn,
      resultStates,
      predictionPill,
      predictionLabel,
      confidenceValue,
      confidenceFill,
      errorMessage,
    });
  });

  /* ---- Reset button ---- */
  if (resetBtn) {
    resetBtn.addEventListener('click', () => {
      resetPredictForm({
        form,
        resultStates,
        confidenceFill,
        confidenceValue,
        predictionPill,
      });
    });
  }

  /* ---- Retry button (after an error) ---- */
  if (retryBtn) {
    retryBtn.addEventListener('click', () => {
      showResultState(resultStates, 'stateIdle');
    });
  }
}

/**
 * Populates the prediction form with a known sample grain's measurements.
 */
function fillSampleValues(form) {
  Object.entries(SAMPLE_GRAIN).forEach(([fieldName, value]) => {
    const input = form.elements.namedItem(fieldName);
    if (input) {
      input.value = value;
      input.dispatchEvent(new Event('input', { bubbles: true }));
    }
  });
}

/**
 * Handles the predict form submission: shows loading state, calls the
 * FastAPI backend, and transitions to success or error state.
 */
async function handlePredictSubmit({
  form,
  submitBtn,
  resultStates,
  predictionPill,
  predictionLabel,
  confidenceValue,
  confidenceFill,
  errorMessage,
}) {
  const formData = new FormData(form);
  const payload = {};
  formData.forEach((value, key) => {
    const numericValue = parseFloat(value);
    payload[key] = Number.isNaN(numericValue) ? value : numericValue;
  });

  setSubmitLoading(submitBtn, true);
  showResultState(resultStates, 'stateLoading');

  try {
    const response = await fetch('/api/predict', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Accept: 'application/json',
      },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      throw new Error(`Prediction request failed with status ${response.status}`);
    }

    const data = await response.json();
    const { label, confidencePercent } = normalizePredictionResponse(data);

    await presentPredictionResult({
      label,
      confidencePercent,
      predictionPill,
      predictionLabel,
      confidenceValue,
      confidenceFill,
    });

    showResultState(resultStates, 'stateSuccess');
  } catch (error) {
    if (errorMessage) {
      errorMessage.textContent =
        'Something went wrong while contacting the prediction service. Please try again.';
    }
    showResultState(resultStates, 'stateError');
  } finally {
    setSubmitLoading(submitBtn, false);
  }
}

/**
 * Normalizes the backend's prediction response into a consistent shape,
 * tolerating either a 0–1 confidence fraction or a 0–100 percentage.
 */
function normalizePredictionResponse(data) {
  const label = data.prediction ?? data.label ?? data.variety ?? 'Unknown';
  let confidence = data.confidence ?? data.probability ?? data.score ?? 0;

  confidence = Number(confidence) || 0;
  const confidencePercent = confidence <= 1 ? confidence * 100 : confidence;

  return { label, confidencePercent: clamp(confidencePercent, 0, 100) };
}

/**
 * Updates the success state UI (pill label/color, confidence bar + value)
 * with an animated confidence fill.
 */
function presentPredictionResult({
  label,
  confidencePercent,
  predictionPill,
  predictionLabel,
  confidenceValue,
  confidenceFill,
}) {
  if (predictionLabel) {
    predictionLabel.textContent = label;
  }

  if (predictionPill) {
    const isOsmancik = String(label).toLowerCase() === 'osmancik';
    predictionPill.classList.toggle('is-osmancik', isOsmancik);
  }

  if (confidenceFill) {
    confidenceFill.style.width = '0%';
  }
  if (confidenceValue) {
    confidenceValue.textContent = '0%';
  }

  return animateValue({
    from: 0,
    to: confidencePercent,
    duration: 900,
    onUpdate: (value) => {
      if (confidenceFill) {
        confidenceFill.style.width = `${value}%`;
      }
      if (confidenceValue) {
        confidenceValue.textContent = `${value.toFixed(1)}%`;
      }
    },
    onComplete: () => {
      if (confidenceValue) {
        confidenceValue.textContent = `${confidencePercent.toFixed(1)}%`;
      }
    },
  });
}

/**
 * Toggles the submit button's loading visual state (spinner + disabled).
 */
function setSubmitLoading(submitBtn, isLoading) {
  if (!submitBtn) return;
  submitBtn.disabled = isLoading;
  submitBtn.classList.toggle('is-loading', isLoading);
}

/**
 * Shows exactly one of the result state panels (idle/loading/success/error)
 * and hides the rest, using the `hidden` attribute already present in HTML.
 */
function showResultState(resultStates, activeKey) {
  Object.entries(resultStates).forEach(([key, el]) => {
    if (!el) return;
    el.hidden = key !== activeKey;
  });
}

/**
 * Resets the predict form and result panel back to their idle state.
 */
function resetPredictForm({ form, resultStates, confidenceFill, confidenceValue, predictionPill }) {
  form.reset();

  if (confidenceFill) {
    confidenceFill.style.width = '0%';
  }
  if (confidenceValue) {
    confidenceValue.textContent = '0%';
  }
  if (predictionPill) {
    predictionPill.classList.remove('is-osmancik');
  }

  showResultState(resultStates, 'stateIdle');
}
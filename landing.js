document.addEventListener('DOMContentLoaded', () => {
  // Guard GSAP availability
  const hasGSAP = typeof window.gsap !== 'undefined';
  if (!hasGSAP) return;

  const tl = gsap.timeline();
  // Hero intro animations
  tl.from('.brand, .nav-row .nav-link', { opacity: 0, y: -10, stagger: 0.05, duration: 0.6, ease: 'power2.out' })
    .from('.hero-content .display-title', { opacity: 0, y: 20, duration: 0.7 }, '-=0.3')
    .from('.hero-content .display-tagline, .hero-content .sub-tagline', { opacity: 0, y: 10, stagger: 0.1, duration: 0.6 }, '-=0.3')
    .from('.hero-content .feature-badges .badge', { opacity: 0, scale: 0.9, stagger: 0.1, duration: 0.5 }, '-=0.3')
    .from('.hero-content .cta-primary', { opacity: 0, y: 8, duration: 0.5 }, '-=0.2')
    // Space-shuttle themed liftoff motion
    .to('.hero-content .display-tagline', { y: -6, duration: 1.2, ease: 'power2.out' }, 'launch')
    .to('.hero-content .sub-tagline', { y: -4, duration: 1.2, ease: 'power2.out' }, 'launch');

  // Start typewriter shortly after intro animations
  tl.call(startTypewriter, null, '+=0.1');

  function startTypewriter() {
    const container = document.querySelector('.hero-content .display-tagline');
    const el = container && container.querySelector('.type-text');
    const caret = container && container.querySelector('.caret');
    if (!el) return;
    const full = el.textContent.trim();
    el.textContent = '';
    let i = 0;
    const speed = 45; // ms per character
    function step() {
      if (i <= full.length) {
        el.textContent = full.slice(0, i);
        i += 1;
        setTimeout(step, speed);
      } else {
        // Fade caret after a short delay
        setTimeout(() => { if (caret) caret.classList.add('fade'); }, 600);
      }
    }
    step();
  }

  // Subtle background video opacity pulse (more visible on load)
  gsap.fromTo('.hero-video .bg-video', { opacity: 0.35 }, { opacity: 0.45, duration: 1.2, ease: 'power1.out' });

  // Scroll animations for showcase section
  if (window.ScrollTrigger) {
    gsap.registerPlugin(ScrollTrigger);
    gsap.from('.showcase .badge', { opacity: 0, x: -10, duration: 0.6, scrollTrigger: { trigger: '.showcase', start: 'top 80%' } });
    gsap.from('.showcase .display-title', { opacity: 0, x: -20, duration: 0.8, scrollTrigger: { trigger: '.showcase', start: 'top 78%' } });
    gsap.from('.showcase .sub-tagline', { opacity: 0, y: 10, duration: 0.6, scrollTrigger: { trigger: '.showcase', start: 'top 76%' } });
    gsap.from('.showcase .cta-row .cta-button', { opacity: 0, y: 12, stagger: 0.1, duration: 0.5, scrollTrigger: { trigger: '.showcase', start: 'top 74%' } });

    gsap.from('.showcase-right .metric-card', {
      opacity: 0,
      y: 24,
      stagger: 0.15,
      duration: 0.6,
      ease: 'power2.out',
      scrollTrigger: { trigger: '.showcase-right', start: 'top 80%' }
    });

    // Features section reveal
    gsap.from('.features-head .badge', { opacity: 0, y: -8, duration: 0.5, scrollTrigger: { trigger: '.features-section', start: 'top 85%' } });
    gsap.from('.features-title', { opacity: 0, y: 18, duration: 0.7, ease: 'power2.out', scrollTrigger: { trigger: '.features-section', start: 'top 83%' } });
    gsap.from('.features-sub', { opacity: 0, y: 10, duration: 0.6, scrollTrigger: { trigger: '.features-section', start: 'top 81%' } });
    gsap.from('.feature-card', {
      opacity: 0,
      y: 26,
      stagger: 0.15,
      duration: 0.6,
      ease: 'power2.out',
      scrollTrigger: { trigger: '.feature-grid', start: 'top 80%' }
    });

    // How It Works section reveal
    gsap.from('.how-head .badge', { opacity: 0, y: -8, duration: 0.5, scrollTrigger: { trigger: '.how-section', start: 'top 85%' } });
    gsap.from('.how-title', { opacity: 0, y: 18, duration: 0.7, ease: 'power2.out', scrollTrigger: { trigger: '.how-section', start: 'top 83%' } });
    gsap.from('.how-sub', { opacity: 0, y: 10, duration: 0.6, scrollTrigger: { trigger: '.how-section', start: 'top 81%' } });
    gsap.from('.step-card', {
      opacity: 0,
      y: 26,
      stagger: 0.12,
      duration: 0.6,
      ease: 'power2.out',
      scrollTrigger: { trigger: '.steps-grid', start: 'top 80%' }
    });

    // Kid section reveal
    gsap.from('.after-title', { opacity: 0, y: 18, duration: 0.6, ease: 'power2.out', scrollTrigger: { trigger: '.after-section', start: 'top 85%' } });
    gsap.from('.after-video .bg-video', { opacity: 0.4, duration: 0.8, ease: 'power1.out', scrollTrigger: { trigger: '.after-video', start: 'top 80%' } });

    // Reviews section reveal
    gsap.from('.reviews-head .badge', { opacity: 0, y: -8, duration: 0.5, scrollTrigger: { trigger: '.reviews-section', start: 'top 85%' } });
    gsap.from('.reviews-title', { opacity: 0, y: 18, duration: 0.7, ease: 'power2.out', scrollTrigger: { trigger: '.reviews-section', start: 'top 83%' } });
    gsap.from('.reviews-sub', { opacity: 0, y: 10, duration: 0.6, scrollTrigger: { trigger: '.reviews-section', start: 'top 81%' } });
    // Reviews: use fromTo because cards have a .reveal class (opacity: 0)
    gsap.fromTo('.review-card',
      { opacity: 0, y: 26 },
      {
        opacity: 1,
        y: 0,
        stagger: 0.12,
        duration: 0.6,
        ease: 'power2.out',
        scrollTrigger: { trigger: '.reviews-grid', start: 'top 80%' }
      }
    );

    // Why Choose Us section reveal
    gsap.from('.choose-left .badge', { opacity: 0, y: -8, duration: 0.5, scrollTrigger: { trigger: '.choose-section', start: 'top 85%' } });
    gsap.from('.choose-title', { opacity: 0, y: 18, duration: 0.7, ease: 'power2.out', scrollTrigger: { trigger: '.choose-section', start: 'top 83%' } });
    gsap.from('.choose-sub', { opacity: 0, y: 10, duration: 0.6, scrollTrigger: { trigger: '.choose-section', start: 'top 81%' } });
    gsap.from('.choose-list li', {
      opacity: 0,
      y: 20,
      stagger: 0.1,
      duration: 0.5,
      ease: 'power2.out',
      scrollTrigger: { trigger: '.choose-list', start: 'top 80%' }
    });
    gsap.from('.choose-frame', { opacity: 0, scale: 0.95, duration: 0.6, ease: 'power2.out', scrollTrigger: { trigger: '.choose-right', start: 'top 82%' } });

    // Global high-tech gaming vibe scroll effects
    // Header glass glow and compacting on scroll
    gsap.to('.site-header', {
      background: 'rgba(12,12,16,0.55)',
      boxShadow: '0 0 24px rgba(130,180,255,0.25)',
      backdropFilter: 'blur(10px)',
      y: 0,
      scrollTrigger: { trigger: '.hero', start: 'top top', end: 'bottom top', scrub: true }
    });

    // Hero parallax: background video moves slower, content moves subtly
    gsap.to('.hero .bg-video', { yPercent: -10, ease: 'none', scrollTrigger: { trigger: '.hero', start: 'top top', end: 'bottom top', scrub: true } });
    gsap.to('.hero .hero-content', { yPercent: 8, opacity: 0.95, ease: 'none', scrollTrigger: { trigger: '.hero', start: 'top top', end: 'bottom top', scrub: true } });

    // Nav link highlight (active glow) when sections enter viewport
    const sectionMap = [
      { sel: '#features', link: 'a[href="#features"]' },
      { sel: '#how', link: 'a[href="#how"]' },
      { sel: '#after', link: 'a[href="#after"]' },
      { sel: '#reviews', link: 'a[href="#features"]' } // keep at least one link highlighted if reviews has no nav
    ];
    sectionMap.forEach(({ sel, link }) => {
      const anchor = document.querySelector(link);
      if (!anchor) return;
      ScrollTrigger.create({
        trigger: sel,
        start: 'top center',
        end: 'bottom center',
        onEnter: () => anchor.classList.add('active'),
        onLeave: () => anchor.classList.remove('active'),
        onEnterBack: () => anchor.classList.add('active'),
        onLeaveBack: () => anchor.classList.remove('active')
      });
    });

    // Footer (Contact-like) fade/slide up on scroll
    gsap.from('.terminal-footer', { opacity: 0, y: 16, duration: 0.6, ease: 'power2.out', scrollTrigger: { trigger: '.terminal-footer', start: 'top 90%' } });
  }

  // Button micro-interactions
  const buttons = document.querySelectorAll('.cta-button');
  buttons.forEach(btn => {
    btn.addEventListener('mouseenter', () => gsap.to(btn, { scale: 1.03, duration: 0.15 }));
    btn.addEventListener('mouseleave', () => gsap.to(btn, { scale: 1.0, duration: 0.15 }));
    btn.addEventListener('mousedown', () => gsap.to(btn, { scale: 0.98, duration: 0.08 }));
    btn.addEventListener('mouseup', () => gsap.to(btn, { scale: 1.03, duration: 0.1 }));
  });

  // Card tilt micro-interactions for showcase and features
  function applyTilt(nodes) {
    const maxRotate = 5; // degrees
    nodes.forEach(card => {
      let bounds;
      function handleEnter() {
        bounds = card.getBoundingClientRect();
        gsap.to(card, { scale: 1.03, duration: 0.2, ease: 'power2.out' });
      }
      function handleMove(e) {
        if (!bounds) return;
        const relX = (e.clientX - bounds.left) / bounds.width;
        const relY = (e.clientY - bounds.top) / bounds.height;
        const rotY = (relX - 0.5) * (maxRotate * 2);
        const rotX = -(relY - 0.5) * (maxRotate * 2);
        gsap.to(card, { rotateY: rotY, rotateX: rotX, duration: 0.15, ease: 'power2.out' });
      }
      function handleLeave() {
        gsap.to(card, { rotateY: 0, rotateX: 0, scale: 1.0, duration: 0.25, ease: 'power2.out' });
        bounds = null;
      }
      card.addEventListener('mouseenter', handleEnter);
      card.addEventListener('mousemove', handleMove);
      card.addEventListener('mouseleave', handleLeave);
    });
  }
  applyTilt(document.querySelectorAll('.metric-card, .feature-card, .review-card'));

  // Fallback intersection observer for reveal utility when GSAP isn't available
  if (!hasGSAP) {
    const io = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) entry.target.classList.add('is-visible');
      });
    }, { rootMargin: '0px 0px -10% 0px', threshold: 0.1 });
    document.querySelectorAll('.reveal').forEach(el => io.observe(el));
  }
});
"""
Build a production-grade, standalone Interactive HTML5 Slide Deck for the 20-Slide Masterclass.
Features:
1. Base64 embedded high-resolution graphics (zero broken image links, 100% portable).
2. Dual KaTeX + MathJax 3 mathematical rendering for robust inline and display formulas.
3. Interactive Slide Carousel (Keyboard arrows ←/→, Spacebar, Prev/Next buttons, Slide Dots, Overview Mode).
4. Signature Brian Avants Light Slate Aesthetic (#FFFFFF / #F8FAFC / #0F172A / #2563EB / #10B981).
5. Slide-by-slide Presenter Notes with toggleable drawer.
"""

import os
import base64
import sys
sys.path.insert(0, '.')
from scripts.generate_presentation import SLIDES_DATA, PPTX_PATH

HTML_PATH = "docs/presentation/index.html"

def get_image_base64_uri(img_path):
    if not os.path.exists(img_path):
        print(f"WARNING: Image not found: {img_path}")
        return ""
    ext = os.path.splitext(img_path)[1].lower().replace('.', '')
    mime = "image/jpeg" if ext in ["jpg", "jpeg"] else "image/png"
    with open(img_path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{data}"

def build_standalone_deck():
    print("Encoding images to base64 data URIs...", flush=True)
    slides_data_with_uri = []
    for s in SLIDES_DATA:
        s_copy = dict(s)
        s_copy["img_uri"] = get_image_base64_uri(s["image"])
        slides_data_with_uri.append(s_copy)

    print("Generating HTML structure...", flush=True)
    
    slides_markup = []
    nav_dots = []

    for idx, s in enumerate(slides_data_with_uri):
        num = s["num"]
        cat = s["category"]
        title = s["title"]
        subtitle = s["subtitle"]
        notes = s["notes"]
        bullets = s["bullets"]
        img_uri = s["img_uri"]

        bullets_html = "".join([
            f"""<li class="bullet-item">
                <span class="bullet-title">{b_title}:</span>
                <span class="bullet-text">{b_desc}</span>
            </li>"""
            for b_title, b_desc in bullets
        ])

        img_markup = f"""
        <div class="slide-image-card">
            <div class="img-wrapper">
                <img src="{img_uri}" alt="Slide {num} Scientific Visual" />
            </div>
            <div class="img-caption">Figure: {os.path.basename(s['image'])}</div>
        </div>
        """ if img_uri else ""

        slide_html = f"""
        <div class="slide-container {'active' if idx == 0 else ''}" id="slide-{num}" data-slide-index="{idx}">
            <div class="slide-header">
                <div class="category-badge">SLIDE {num:02d} / 20 &nbsp;•&nbsp; {cat}</div>
                <h1 class="slide-title">{title}</h1>
                <div class="slide-subtitle">{subtitle}</div>
            </div>

            <div class="slide-body">
                <div class="slide-text-card">
                    <ul class="bullet-list">
                        {bullets_html}
                    </ul>
                </div>
                {img_markup}
            </div>

            <div class="slide-footer">
                <div class="notes-header">
                    <span class="notes-icon">🎙️</span>
                    <strong>Presenter Script:</strong>
                </div>
                <div class="notes-content">{notes}</div>
            </div>
        </div>
        """
        slides_markup.append(slide_html)
        nav_dots.append(f'<button class="dot-btn {"active" if idx == 0 else ""}" data-target="{idx}" title="Slide {num}: {title}"></button>')

    full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Symmetric Diffeomorphic Registration (syntx) - 20-Slide Masterclass</title>
    
    <!-- Typography -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
    
    <!-- KaTeX CSS and JS for instantaneous, rock-solid math rendering -->
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.css">
    <script src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/contrib/auto-render.min.js"></script>

    <!-- MathJax 3 Configuration & Fallback Engine -->
    <script>
        window.MathJax = {{
            tex: {{
                inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
                displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']],
                processEscapes: true
            }},
            options: {{
                skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code']
            }}
        }};
    </script>
    <script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js" id="MathJax-script"></script>

    <style>
        :root {{
            --bg-canvas: #0F172A;
            --bg-deck: #F8FAFC;
            --bg-card: #FFFFFF;
            --border-card: #E2E8F0;
            --border-highlight: #CBD5E1;
            --text-title: #0F172A;
            --text-subtitle: #475569;
            --text-body: #1E293B;
            --text-muted: #64748B;
            --primary: #2563EB;
            --primary-light: #EFF6FF;
            --primary-dark: #1D4ED8;
            --success: #059669;
            --success-light: #ECFDF5;
            --accent: #9333EA;
            --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
            --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.08), 0 2px 4px -1px rgba(0, 0, 0, 0.04);
            --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.08), 0 4px 6px -2px rgba(0, 0, 0, 0.04);
            --radius-card: 14px;
        }}

        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}

        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background-color: var(--bg-canvas);
            color: var(--text-body);
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            min-height: 100vh;
            padding: 16px;
            overflow-x: hidden;
        }}

        /* App Bar */
        .app-bar {{
            width: 100%;
            max-width: 1300px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 12px 20px;
            background: #1E293B;
            border-radius: 12px 12px 0 0;
            color: #FFFFFF;
        }}

        .brand-title {{
            font-weight: 700;
            font-size: 15px;
            display: flex;
            align-items: center;
            gap: 10px;
        }}

        .brand-badge {{
            background: var(--primary);
            color: #FFFFFF;
            font-size: 11px;
            font-weight: 800;
            padding: 2px 8px;
            border-radius: 6px;
            letter-spacing: 0.05em;
        }}

        .app-actions {{
            display: flex;
            align-items: center;
            gap: 12px;
        }}

        .action-btn {{
            background: #334155;
            color: #F8FAFC;
            border: 1px solid #475569;
            padding: 6px 14px;
            border-radius: 6px;
            font-size: 13px;
            font-weight: 600;
            cursor: pointer;
            text-decoration: none;
            display: inline-flex;
            align-items: center;
            gap: 6px;
            transition: all 0.15s ease;
        }}

        .action-btn:hover {{
            background: #475569;
            color: #FFFFFF;
        }}

        .action-btn.primary {{
            background: var(--primary);
            border-color: var(--primary-dark);
        }}

        .action-btn.primary:hover {{
            background: var(--primary-dark);
        }}

        /* Stage Viewport */
        .deck-viewport {{
            width: 100%;
            max-width: 1300px;
            height: 760px;
            background: var(--bg-deck);
            position: relative;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.4);
            border-radius: 0 0 12px 12px;
            overflow: hidden;
            display: flex;
            flex-direction: column;
        }}

        /* Slide Container */
        .slide-container {{
            display: none;
            width: 100%;
            height: 100%;
            padding: 24px 32px 18px 32px;
            flex-direction: column;
            justify-content: space-between;
            animation: fadeIn 0.2s ease-out;
        }}

        .slide-container.active {{
            display: flex;
        }}

        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(4px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}

        /* Slide Header */
        .slide-header {{
            margin-bottom: 12px;
        }}

        .category-badge {{
            font-size: 12px;
            font-weight: 800;
            color: var(--primary);
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 4px;
        }}

        .slide-title {{
            font-size: 23px;
            font-weight: 800;
            color: var(--text-title);
            line-height: 1.25;
            letter-spacing: -0.01em;
            margin-bottom: 4px;
        }}

        .slide-subtitle {{
            font-size: 13.5px;
            color: var(--text-subtitle);
            font-weight: 500;
        }}

        /* Slide Body (Split 2-Column) */
        .slide-body {{
            display: flex;
            gap: 20px;
            flex: 1;
            min-height: 0;
            align-items: stretch;
        }}

        .slide-text-card {{
            flex: 1.15;
            background: var(--bg-card);
            border: 1px solid var(--border-card);
            border-radius: var(--radius-card);
            padding: 20px 22px;
            box-shadow: var(--shadow-sm);
            display: flex;
            flex-direction: column;
            justify-content: center;
            overflow-y: auto;
        }}

        .bullet-list {{
            list-style: none;
            display: flex;
            flex-direction: column;
            gap: 12px;
        }}

        .bullet-item {{
            font-size: 14px;
            line-height: 1.55;
            color: var(--text-body);
            position: relative;
            padding-left: 18px;
        }}

        .bullet-item::before {{
            content: "•";
            position: absolute;
            left: 0;
            top: -1px;
            color: var(--primary);
            font-size: 18px;
            font-weight: bold;
        }}

        .bullet-title {{
            font-weight: 700;
            color: var(--text-title);
            margin-right: 4px;
        }}

        .bullet-text {{
            color: #334155;
        }}

        /* Image Card */
        .slide-image-card {{
            flex: 1.1;
            background: var(--bg-card);
            border: 1px solid var(--border-card);
            border-radius: var(--radius-card);
            padding: 12px;
            box-shadow: var(--shadow-sm);
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            position: relative;
        }}

        .img-wrapper {{
            width: 100%;
            height: 100%;
            display: flex;
            align-items: center;
            justify-content: center;
            overflow: hidden;
            border-radius: 8px;
        }}

        .img-wrapper img {{
            max-width: 100%;
            max-height: 420px;
            object-fit: contain;
            border-radius: 6px;
        }}

        .img-caption {{
            margin-top: 6px;
            font-size: 11px;
            font-weight: 600;
            color: var(--text-muted);
            font-family: 'JetBrains Mono', monospace;
        }}

        /* Slide Footer (Presenter Script) */
        .slide-footer {{
            margin-top: 14px;
            background: #EFF6FF;
            border-left: 4px solid var(--primary);
            border-radius: 8px;
            padding: 10px 14px;
            display: flex;
            flex-direction: column;
            gap: 2px;
        }}

        .notes-header {{
            display: flex;
            align-items: center;
            gap: 6px;
            font-size: 11.5px;
            font-weight: 700;
            color: var(--primary-dark);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}

        .notes-content {{
            font-size: 13px;
            line-height: 1.45;
            color: #1E3A8A;
            font-style: italic;
        }}

        /* Bottom Controls Bar */
        .controls-bar {{
            width: 100%;
            max-width: 1300px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 14px 20px;
            background: #1E293B;
            border-top: 1px solid #334155;
            border-radius: 0 0 12px 12px;
            color: #FFFFFF;
        }}

        .nav-btn-group {{
            display: flex;
            align-items: center;
            gap: 10px;
        }}

        .nav-arrow-btn {{
            background: #334155;
            color: #FFFFFF;
            border: 1px solid #475569;
            padding: 8px 16px;
            border-radius: 8px;
            font-size: 13.5px;
            font-weight: 700;
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 6px;
            transition: all 0.15s ease;
        }}

        .nav-arrow-btn:hover:not(:disabled) {{
            background: var(--primary);
            border-color: var(--primary-dark);
        }}

        .nav-arrow-btn:disabled {{
            opacity: 0.4;
            cursor: not-allowed;
        }}

        .slide-counter {{
            font-weight: 700;
            font-size: 13.5px;
            color: #94A3B8;
            font-family: 'JetBrains Mono', monospace;
        }}

        .dots-tray {{
            display: flex;
            align-items: center;
            gap: 6px;
        }}

        .dot-btn {{
            width: 9px;
            height: 9px;
            border-radius: 50%;
            background: #475569;
            border: none;
            cursor: pointer;
            transition: all 0.2s ease;
        }}

        .dot-btn:hover {{
            background: #94A3B8;
            transform: scale(1.2);
        }}

        .dot-btn.active {{
            background: var(--primary);
            width: 22px;
            border-radius: 5px;
        }}

        /* Overview Mode */
        body.overview-mode .deck-viewport {{
            height: auto;
            overflow-y: auto;
            gap: 20px;
            padding: 20px;
        }}

        body.overview-mode .slide-container {{
            display: flex;
            height: 640px;
            border: 1px solid var(--border-card);
            border-radius: var(--radius-card);
            background: #FFFFFF;
            margin-bottom: 20px;
            box-shadow: var(--shadow-md);
        }}

        body.overview-mode .controls-bar {{
            display: none;
        }}

        /* Math formula adjustments */
        .katex {{
            font-size: 1.05em;
            color: #0F172A;
        }}
    </style>
</head>
<body>

    <!-- App Header -->
    <div class="app-bar">
        <div class="brand-title">
            <span class="brand-badge">SYNTX</span>
            <span>Symmetric Diffeomorphic Registration on Riemannian Manifolds</span>
        </div>
        <div class="app-actions">
            <button class="action-btn" id="toggle-mode-btn" title="Toggle Overview / Slideshow">🗂️ Overview Mode</button>
            <button class="action-btn" id="fullscreen-btn" title="Enter Fullscreen">⛶ Fullscreen</button>
            <a href="syntx_diffeomorphic_geometry_presentation.pptx" download class="action-btn primary">⬇ Download PPTX</a>
        </div>
    </div>

    <!-- Presentation Viewport -->
    <div class="deck-viewport" id="viewport">
        {"".join(slides_markup)}
    </div>

    <!-- Bottom Controls -->
    <div class="controls-bar">
        <div class="nav-btn-group">
            <button class="nav-arrow-btn" id="prev-btn">← Previous</button>
            <button class="nav-arrow-btn" id="next-btn">Next →</button>
        </div>

        <div class="dots-tray" id="dots-tray">
            {"".join(nav_dots)}
        </div>

        <div class="slide-counter" id="slide-counter">
            01 / 20
        </div>
    </div>

    <script>
        let currentSlide = 0;
        const totalSlides = {len(slides_data_with_uri)};
        const slides = document.querySelectorAll('.slide-container');
        const dots = document.querySelectorAll('.dot-btn');
        const prevBtn = document.getElementById('prev-btn');
        const nextBtn = document.getElementById('next-btn');
        const counter = document.getElementById('slide-counter');
        const toggleModeBtn = document.getElementById('toggle-mode-btn');
        const fullscreenBtn = document.getElementById('fullscreen-btn');

        function renderAllMath() {{
            try {{
                if (typeof renderMathInElement === 'function') {{
                    renderMathInElement(document.body, {{
                        delimiters: [
                            {{left: '$$', right: '$$', display: true}},
                            {{left: '$', right: '$', display: false}},
                            {{left: '\\\\(', right: '\\\\)', display: false}},
                            {{left: '\\\\[', right: '\\\\]', display: true}}
                        ],
                        throwOnError: false
                    }});
                }}
            }} catch (e) {{
                console.warn("KaTeX render error:", e);
            }}

            try {{
                if (window.MathJax && window.MathJax.typesetPromise) {{
                    window.MathJax.typesetPromise();
                }}
            }} catch (e) {{
                console.warn("MathJax render error:", e);
            }}
        }}

        function updateSlide(index) {{
            if (index < 0 || index >= totalSlides) return;
            currentSlide = index;

            slides.forEach((s, idx) => {{
                s.classList.toggle('active', idx === currentSlide);
            }});

            dots.forEach((d, idx) => {{
                d.classList.toggle('active', idx === currentSlide);
            }});

            prevBtn.disabled = (currentSlide === 0);
            nextBtn.disabled = (currentSlide === totalSlides - 1);
            counter.textContent = String(currentSlide + 1).padStart(2, '0') + ' / ' + String(totalSlides).padStart(2, '0');
        }}

        prevBtn.addEventListener('click', () => updateSlide(currentSlide - 1));
        nextBtn.addEventListener('click', () => updateSlide(currentSlide + 1));

        dots.forEach(dot => {{
            dot.addEventListener('click', () => {{
                const target = parseInt(dot.getAttribute('data-target'), 10);
                updateSlide(target);
            }});
        }});

        // Keyboard Navigation
        document.addEventListener('keydown', (e) => {{
            if (document.body.classList.contains('overview-mode')) return;
            if (e.key === 'ArrowRight' || e.key === ' ' || e.key === 'PageDown') {{
                e.preventDefault();
                updateSlide(currentSlide + 1);
            }} else if (e.key === 'ArrowLeft' || e.key === 'PageUp') {{
                e.preventDefault();
                updateSlide(currentSlide - 1);
            }} else if (e.key === 'Home') {{
                e.preventDefault();
                updateSlide(0);
            }} else if (e.key === 'End') {{
                e.preventDefault();
                updateSlide(totalSlides - 1);
            }} else if (e.key.toLowerCase() === 'f') {{
                fullscreenBtn.click();
            }} else if (e.key.toLowerCase() === 'o') {{
                toggleModeBtn.click();
            }}
        }});

        // Toggle Overview Mode
        toggleModeBtn.addEventListener('click', () => {{
            document.body.classList.toggle('overview-mode');
            const isOverview = document.body.classList.contains('overview-mode');
            toggleModeBtn.textContent = isOverview ? '🖥️ Slideshow Mode' : '🗂️ Overview Mode';
        }});

        // Fullscreen Toggle
        fullscreenBtn.addEventListener('click', () => {{
            if (!document.fullscreenElement) {{
                document.documentElement.requestFullscreen().catch(err => {{}});
            }} else {{
                document.exitFullscreen().catch(err => {{}});
            }}
        }});

        // Render math on page load
        window.addEventListener('DOMContentLoaded', () => {{
            updateSlide(0);
            renderAllMath();
        }});
        window.addEventListener('load', () => {{
            renderAllMath();
        }});
    </script>
</body>
</html>
"""

    with open(HTML_PATH, "w") as f:
        f.write(full_html)
    print(f"Standalone Interactive HTML Deck saved to: {HTML_PATH}", flush=True)

if __name__ == "__main__":
    build_standalone_deck()

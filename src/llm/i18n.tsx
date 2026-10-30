'use client';

import React, { useEffect, useState } from 'react';

// Language selection for the LLM walkthrough. Precedence: ?lang= URL param, then localStorage.
// Switching reloads the page, so text only needs to be picked at render time (no reactivity needed).

export type Lang = 'en' | 'zh-TW';

const storageKey = 'llm-viz-lang';

function parseLang(s: string | null | undefined): Lang | null {
    return s === 'en' || s === 'zh-TW' ? s : null;
}

let currLang: Lang | null = null;

export function getLang(): Lang {
    if (typeof window === 'undefined') {
        return 'en';
    }
    if (!currLang) {
        let fromUrl = parseLang(new URLSearchParams(window.location.search).get('lang'));
        let fromStorage: Lang | null = null;
        try { fromStorage = parseLang(window.localStorage.getItem(storageKey)); } catch { }
        currLang = fromUrl ?? fromStorage ?? 'en';
    }
    return currLang;
}

export function isZh() {
    return getLang() === 'zh-TW';
}

/** Pick a plain string by language. For commentary templates, use `isZh() ? commentary(wt)`...` : commentary(wt)`...``. */
export function tr(en: string, zh: string) {
    return isZh() ? zh : en;
}

export function setLang(lang: Lang) {
    try { window.localStorage.setItem(storageKey, lang); } catch { }
    let url = new URL(window.location.href);
    url.searchParams.delete('lang');
    window.location.href = url.toString();
}

export const LangButton: React.FC = () => {
    // Render nothing until mounted, so server & client markup match.
    let [lang, setLangState] = useState<Lang | null>(null);
    useEffect(() => setLangState(getLang()), []);

    if (!lang) {
        return null;
    }

    let next: Lang = lang === 'en' ? 'zh-TW' : 'en';
    return <div className="ml-3 cursor-pointer hover:underline" onClick={() => setLang(next)}>
        {next === 'en' ? 'English' : '繁體中文'}
    </div>;
};

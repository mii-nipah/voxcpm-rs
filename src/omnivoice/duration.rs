use unicode_general_category::{get_general_category, GeneralCategory};

#[derive(Debug)]
pub struct RuleDurationEstimator {
    ranges: &'static [(u32, &'static str)],
}

impl Default for RuleDurationEstimator {
    fn default() -> Self {
        Self::new()
    }
}

impl RuleDurationEstimator {
    pub fn new() -> Self {
        // Ranges sorted by the codepoint end value
        let ranges: &'static [(u32, &'static str)] = &[
            (0x02AF, "latin"),
            (0x03FF, "greek"),
            (0x052F, "cyrillic"),
            (0x058F, "armenian"),
            (0x05FF, "hebrew"),
            (0x077F, "arabic"),
            (0x089F, "arabic"),
            (0x08FF, "arabic"),
            (0x097F, "indic"),
            (0x09FF, "indic"),
            (0x0A7F, "indic"),
            (0x0AFF, "indic"),
            (0x0B7F, "indic"),
            (0x0BFF, "indic"),
            (0x0C7F, "indic"),
            (0x0CFF, "indic"),
            (0x0D7F, "indic"),
            (0x0DFF, "indic"),
            (0x0EFF, "thai_lao"),
            (0x0FFF, "indic"),
            (0x109F, "khmer_myanmar"),
            (0x10FF, "georgian"),
            (0x11FF, "hangul"),
            (0x137F, "ethiopic"),
            (0x139F, "ethiopic"),
            (0x13FF, "default"),
            (0x167F, "default"),
            (0x169F, "default"),
            (0x16FF, "default"),
            (0x171F, "default"),
            (0x173F, "default"),
            (0x175F, "default"),
            (0x177F, "default"),
            (0x17FF, "khmer_myanmar"),
            (0x18AF, "default"),
            (0x18FF, "default"),
            (0x194F, "indic"),
            (0x19DF, "indic"),
            (0x19FF, "khmer_myanmar"),
            (0x1A1F, "indic"),
            (0x1AAF, "indic"),
            (0x1B7F, "indic"),
            (0x1BBF, "indic"),
            (0x1BFF, "indic"),
            (0x1C4F, "indic"),
            (0x1C7F, "indic"),
            (0x1C8F, "cyrillic"),
            (0x1CBF, "georgian"),
            (0x1CCF, "indic"),
            (0x1CFF, "indic"),
            (0x1D7F, "latin"),
            (0x1DBF, "latin"),
            (0x1DFF, "default"),
            (0x1EFF, "latin"),
            (0x309F, "kana"),
            (0x30FF, "kana"),
            (0x312F, "cjk"),
            (0x318F, "hangul"),
            (0x9FFF, "cjk"),
            (0xA4CF, "yi"),
            (0xA4FF, "default"),
            (0xA63F, "default"),
            (0xA69F, "cyrillic"),
            (0xA6FF, "default"),
            (0xA7FF, "latin"),
            (0xA82F, "indic"),
            (0xA87F, "default"),
            (0xA8DF, "indic"),
            (0xA8FF, "indic"),
            (0xA92F, "indic"),
            (0xA95F, "indic"),
            (0xA97F, "hangul"),
            (0xA9DF, "indic"),
            (0xA9FF, "khmer_myanmar"),
            (0xAA5F, "indic"),
            (0xAA7F, "khmer_myanmar"),
            (0xAADF, "indic"),
            (0xAAFF, "indic"),
            (0xAB2F, "ethiopic"),
            (0xAB6F, "latin"),
            (0xABBF, "default"),
            (0xABFF, "indic"),
            (0xD7AF, "hangul"),
            (0xFAFF, "cjk"),
            (0xFDFF, "arabic"),
            (0xFE6F, "default"),
            (0xFEFF, "arabic"),
            (0xFFEF, "latin"),
        ];
        Self { ranges }
    }

    pub fn get_char_weight(&self, c: char) -> f32 {
        let code = c as u32;

        // Latin basic optimization
        if (65..=90).contains(&code) || (97..=122).contains(&code) {
            return 1.0; // latin weight
        }
        if code == 32 {
            return 0.2; // space weight
        }
        if code == 0x0640 {
            return 0.0; // arabic Tatweel -> mark weight
        }

        let cat = get_general_category(c);
        match cat {
            GeneralCategory::NonspacingMark
            | GeneralCategory::SpacingMark
            | GeneralCategory::EnclosingMark => return 0.0,

            GeneralCategory::ConnectorPunctuation
            | GeneralCategory::DashPunctuation
            | GeneralCategory::OpenPunctuation
            | GeneralCategory::ClosePunctuation
            | GeneralCategory::InitialPunctuation
            | GeneralCategory::FinalPunctuation
            | GeneralCategory::OtherPunctuation
            | GeneralCategory::MathSymbol
            | GeneralCategory::CurrencySymbol
            | GeneralCategory::ModifierSymbol
            | GeneralCategory::OtherSymbol => return 0.5,

            GeneralCategory::SpaceSeparator
            | GeneralCategory::LineSeparator
            | GeneralCategory::ParagraphSeparator => return 0.2,

            GeneralCategory::DecimalNumber
            | GeneralCategory::LetterNumber
            | GeneralCategory::OtherNumber => return 3.5,

            _ => {}
        }

        // Binary search for script block ranges
        match self.ranges.binary_search_by_key(&code, |&(end, _)| end) {
            Ok(idx) => {
                let script = self.ranges[idx].1;
                self.get_script_weight(script)
            }
            Err(idx) => {
                if idx < self.ranges.len() {
                    let script = self.ranges[idx].1;
                    self.get_script_weight(script)
                } else if code > 0x20000 {
                    3.0 // cjk weight
                } else {
                    1.0 // default weight
                }
            }
        }
    }

    fn get_script_weight(&self, script: &str) -> f32 {
        match script {
            "cjk" => 3.0,
            "hangul" => 2.5,
            "kana" => 2.2,
            "ethiopic" => 3.0,
            "yi" => 3.0,
            "indic" => 1.8,
            "thai_lao" => 1.5,
            "khmer_myanmar" => 1.8,
            "arabic" => 1.5,
            "hebrew" => 1.5,
            "latin" => 1.0,
            "cyrillic" => 1.0,
            "greek" => 1.0,
            "armenian" => 1.0,
            "georgian" => 1.0,
            "punctuation" => 0.5,
            "space" => 0.2,
            "digit" => 3.5,
            "mark" => 0.0,
            _ => 1.0,
        }
    }

    pub fn calculate_total_weight(&self, text: &str) -> f32 {
        text.chars().map(|c| self.get_char_weight(c)).sum()
    }

    pub fn estimate_duration(
        &self,
        target_text: &str,
        ref_text: &str,
        ref_duration: f32,
        low_threshold: Option<f32>,
        boost_strength: f32,
    ) -> f32 {
        if ref_duration <= 0.0 || ref_text.is_empty() {
            return 0.0;
        }

        let ref_weight = self.calculate_total_weight(ref_text);
        if ref_weight == 0.0 {
            return 0.0;
        }

        let speed_factor = ref_weight / ref_duration;
        let target_weight = self.calculate_total_weight(target_text);

        let estimated_duration = target_weight / speed_factor;
        if let Some(low) = low_threshold {
            if estimated_duration < low {
                let alpha = 1.0 / boost_strength;
                return low * (estimated_duration / low).powf(alpha);
            }
        }
        estimated_duration
    }
}

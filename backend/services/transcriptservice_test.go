package services

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"arguehub/models"
	"google.golang.org/genai"
)

const validJudgeResult = `{
  "opening_statement": {
    "for": {"score": 8, "reason": "Clear claim with relevant reasoning."},
    "against": {"score": 7, "reason": "Relevant position with limited support."}
  },
  "cross_examination_questions": {
    "for": {"score": 6, "reason": "The question tests an important assumption."},
    "against": {"score": 5, "reason": "The question is only partly relevant."}
  },
  "cross_examination_answers": {
    "for": {"score": 7, "reason": "The answer addresses the challenge."},
    "against": {"score": 6, "reason": "The answer leaves part of the challenge unresolved."}
  },
  "closing": {
    "for": {"score": 8, "reason": "The closing weighs the central impacts."},
    "against": {"score": 7, "reason": "The closing summarizes the opposition."}
  },
  "total": {"for": 0, "against": 0},
  "verdict": {
    "winner": "for",
    "reason": "The For side better supports and weighs its central claim.",
    "congratulations": "The For side wins.",
    "opponent_analysis": "The Against side should answer the key impact comparison."
  }
}`

func TestJudgeDebateHumanVsHumanReportsFailures(t *testing.T) {
	originalClient := geminiClient
	t.Cleanup(func() { geminiClient = originalClient })

	for _, test := range []struct {
		name       string
		statusCode int
		response   string
		wantError  string
		wantCalls  int32
	}{
		{"quota exceeded", http.StatusTooManyRequests, `{"error":{"code":429,"message":"Judge quota exhausted","status":"RESOURCE_EXHAUSTED"}}`, "RESOURCE_EXHAUSTED", 1},
		{"model unavailable", http.StatusNotFound, `{"error":{"code":404,"message":"Model unavailable","status":"NOT_FOUND"}}`, "NOT_FOUND", 1},
		{"permission denied", http.StatusForbidden, `{"error":{"code":403,"message":"API access denied","status":"PERMISSION_DENIED"}}`, "PERMISSION_DENIED", 1},
		{"service unavailable", http.StatusServiceUnavailable, `{"error":{"code":503,"message":"This model is currently experiencing high demand","status":"UNAVAILABLE"}}`, "UNAVAILABLE", 3},
		{"empty response", http.StatusOK, `{"candidates":[]}`, "empty judgment", 1},
	} {
		t.Run(test.name, func(t *testing.T) {
			var requests atomic.Int32
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				requests.Add(1)
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(test.statusCode)
				fmt.Fprint(w, test.response)
			}))
			defer server.Close()
			var err error
			geminiClient, err = genai.NewClient(context.Background(), &genai.ClientConfig{
				APIKey: "test-key", Backend: genai.BackendGeminiAPI,
				HTTPOptions: genai.HTTPOptions{BaseURL: server.URL},
			})
			if err != nil {
				t.Fatal(err)
			}
			result, err := JudgeDebateHumanVsHuman(map[string]string{
				"openingFor": "Alex: Support the motion.", "openingAgainst": "Sam: Oppose the motion.",
			})
			if result != "" || err == nil || !strings.Contains(err.Error(), test.wantError) {
				t.Fatalf("judging = (%q, %v), want empty result and error containing %q", result, err, test.wantError)
			}
			if got := requests.Load(); got != test.wantCalls {
				t.Errorf("Gemini requests = %d, want %d", got, test.wantCalls)
			}
		})
	}
}

func TestJudgeDebateHumanVsHumanSendsAllRounds(t *testing.T) {
	originalClient := geminiClient
	t.Cleanup(func() { geminiClient = originalClient })
	transcripts := map[string]string{
		"topic": "The debate motion", "team_context": "Format: 2-vs-2 team debate.",
		"openingFor": "Alex: Opening argument.", "openingAgainst": "Sam: Opening rebuttal.",
		"crossForQuestion": "Alex: First question?", "crossAgainstAnswer": "Sam: First answer.",
		"crossAgainstQuestion": "Sam: Second question?", "crossForAnswer": "Alex: Second answer.",
		"closingFor": "Alex: Final argument.", "closingAgainst": "Sam: Final rebuttal.",
	}
	var requests atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost || !strings.HasSuffix(r.URL.Path, "/models/"+defaultGeminiModel+":generateContent") {
			t.Errorf("unexpected Gemini request: %s %s", r.Method, r.URL.Path)
		}
		body, err := io.ReadAll(r.Body)
		if err != nil {
			t.Error(err)
		}
		for phase, text := range transcripts {
			if !strings.Contains(string(body), text) {
				t.Errorf("Gemini request is missing %s transcript or context", phase)
			}
		}
		w.Header().Set("Content-Type", "application/json")
		if requests.Add(1) == 1 {
			w.WriteHeader(http.StatusServiceUnavailable)
			fmt.Fprint(w, `{"error":{"code":503,"message":"Temporary high demand","status":"UNAVAILABLE"}}`)
			return
		}
		json.NewEncoder(w).Encode(map[string]interface{}{
			"candidates": []interface{}{map[string]interface{}{
				"content": map[string]interface{}{
					"parts": []interface{}{map[string]string{"text": "Here is the judgment: " + validJudgeResult}},
				},
				"finishReason": "STOP",
			}},
		})
	}))
	defer server.Close()
	var err error
	geminiClient, err = genai.NewClient(context.Background(), &genai.ClientConfig{
		APIKey: "test-key", Backend: genai.BackendGeminiAPI,
		HTTPOptions: genai.HTTPOptions{BaseURL: server.URL},
	})
	if err != nil {
		t.Fatal(err)
	}
	result, err := JudgeDebateHumanVsHuman(transcripts)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := normalizeJudgeResult(result); err != nil {
		t.Fatalf("successful Gemini judgment failed normalization: %v", err)
	}
	if got := requests.Load(); got != 2 {
		t.Errorf("Gemini requests = %d, want 2 after transient 503", got)
	}
}

func TestGenerateJudgeModelTextStopsAtDeadline(t *testing.T) {
	originalClient := geminiClient
	t.Cleanup(func() { geminiClient = originalClient })
	var requests atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requests.Add(1)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusServiceUnavailable)
		fmt.Fprint(w, `{"error":{"code":503,"message":"Temporary high demand","status":"UNAVAILABLE"}}`)
	}))
	defer server.Close()
	var err error
	geminiClient, err = genai.NewClient(context.Background(), &genai.ClientConfig{
		APIKey: "test-key", Backend: genai.BackendGeminiAPI,
		HTTPOptions: genai.HTTPOptions{BaseURL: server.URL},
	})
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 300*time.Millisecond)
	defer cancel()
	if _, err := generateJudgeModelText(ctx, "Test judgment"); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("judging error = %v, want deadline exceeded", err)
	}
	if got := requests.Load(); got != 1 {
		t.Errorf("Gemini requests = %d, want 1 before deadline expires during backoff", got)
	}
}

func TestJudgeDebateHumanVsHumanPreconditions(t *testing.T) {
	originalClient := geminiClient
	t.Cleanup(func() { geminiClient = originalClient })
	geminiClient = nil
	if result, err := JudgeDebateHumanVsHuman(nil); result != "" || err == nil || !strings.Contains(err.Error(), "not initialized") {
		t.Fatalf("uninitialized judge = (%q, %v), want initialization error", result, err)
	}
	geminiClient = &genai.Client{}
	for _, phase := range []string{"openingFor", "openingAgainst"} {
		result, err := JudgeDebateHumanVsHuman(map[string]string{phase: "One side spoke."})
		if result != "" || err == nil || !strings.Contains(err.Error(), "both sides") {
			t.Fatalf("one-sided debate = (%q, %v), want missing transcript error", result, err)
		}
	}
}

func TestNormalizeJudgeResult(t *testing.T) {
	normalized, err := normalizeJudgeResult("```json\n" + validJudgeResult + "\n```")
	if err != nil {
		t.Fatalf("normalizeJudgeResult() error = %v", err)
	}

	var result struct {
		Total struct {
			For     int `json:"for"`
			Against int `json:"against"`
		} `json:"total"`
		Verdict struct {
			Winner string `json:"winner"`
		} `json:"verdict"`
	}
	if err := json.Unmarshal([]byte(normalized), &result); err != nil {
		t.Fatalf("normalized result is not JSON: %v", err)
	}
	if result.Total.For != 29 || result.Total.Against != 25 {
		t.Errorf("normalized totals = (%d, %d), want (29, 25)", result.Total.For, result.Total.Against)
	}
	if result.Verdict.Winner != "For" {
		t.Errorf("normalized winner = %q, want For", result.Verdict.Winner)
	}
}

func TestNormalizeJudgeResultRejectsInvalidJudgments(t *testing.T) {
	tests := []struct {
		name     string
		response string
	}{
		{name: "non-JSON model failure", response: "Unable to judge."},
		{name: "out-of-range score", response: strings.Replace(validJudgeResult, `"score": 8`, `"score": 11`, 1)},
		{name: "missing score explanation", response: strings.Replace(validJudgeResult, `"reason": "Clear claim with relevant reasoning."`, `"reason": ""`, 1)},
		{name: "invalid winner", response: strings.Replace(validJudgeResult, `"winner": "for"`, `"winner": "maybe"`, 1)},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if _, err := normalizeJudgeResult(test.response); err == nil {
				t.Fatal("normalizeJudgeResult() error = nil, want invalid judgment error")
			}
		})
	}
}

func TestMergeTeamTranscriptsCombinesMembersWithoutDuplicateLines(t *testing.T) {
	merged := mergeTeamTranscripts(
		map[string]string{
			"openingFor": "Alex: First argument\nNo response",
			"closingFor": "No response",
		},
		map[string]string{
			"openingFor": "Alex: First argument\nSam: Supporting argument",
			"closingFor": "Sam: Final summary",
		},
	)

	if got, want := merged["openingFor"], "Alex: First argument\nSam: Supporting argument"; got != want {
		t.Errorf("openingFor = %q, want %q", got, want)
	}
	if got, want := merged["closingFor"], "Sam: Final summary"; got != want {
		t.Errorf("closingFor = %q, want %q", got, want)
	}
}

func TestMergeTeamTranscriptsKeepsNoResponseWhenNobodySpoke(t *testing.T) {
	merged := mergeTeamTranscripts(
		map[string]string{"openingFor": "No response"},
		map[string]string{"openingFor": " No response "},
	)
	if got := merged["openingFor"]; got != "No response" {
		t.Errorf("openingFor = %q, want %q", got, "No response")
	}
}

func TestTeamDebateSubmittersUseParticipantsRecordedAtStart(t *testing.T) {
	debate := &models.TeamDebate{
		Team1Stance: "for",
		Team1Members: []models.TeamMember{
			{Email: "new-member@example.com"},
		},
		Participants: []models.TeamDebateParticipant{
			{Email: "participant@example.com", Role: "for"},
			{Email: "opponent@example.com", Role: "against"},
		},
	}

	submitters, err := teamDebateSubmitters(debate, "for")
	if err != nil {
		t.Fatalf("teamDebateSubmitters() error = %v", err)
	}
	if len(submitters) != 1 || submitters[0] != "participant@example.com" {
		t.Fatalf("teamDebateSubmitters() = %v, want [participant@example.com]", submitters)
	}
	if !teamDebateHasSubmitter(debate, "for", "PARTICIPANT@example.com") {
		t.Fatal("teamDebateHasSubmitter() rejected a recorded participant")
	}
	if teamDebateHasSubmitter(debate, "for", "new-member@example.com") {
		t.Fatal("teamDebateHasSubmitter() accepted a member added after the debate started")
	}
}

func TestTeamDebateSubmittersFallBackToRosterForLegacyDebates(t *testing.T) {
	debate := &models.TeamDebate{
		Team1Stance: "for",
		Team1Members: []models.TeamMember{
			{Email: "legacy-member@example.com"},
		},
	}

	submitters, err := teamDebateSubmitters(debate, "for")
	if err != nil {
		t.Fatalf("teamDebateSubmitters() error = %v", err)
	}
	if len(submitters) != 1 || submitters[0] != "legacy-member@example.com" {
		t.Fatalf("teamDebateSubmitters() = %v, want [legacy-member@example.com]", submitters)
	}
}

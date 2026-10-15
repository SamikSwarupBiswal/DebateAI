package services

import (
	"context"
	"errors"
	"net/http"
	"strings"
	"time"

	"google.golang.org/genai"
)

const defaultGeminiModel = "gemini-3.6-flash"

func initGemini(apiKey string) (*genai.Client, error) {
	config := &genai.ClientConfig{}
	if apiKey != "" {
		config.APIKey = apiKey
	}
	return genai.NewClient(context.Background(), config)
}

func generateModelText(ctx context.Context, modelName, prompt string) (string, error) {
	if geminiClient == nil {
		return "", errors.New("gemini client not initialized")
	}

	config := &genai.GenerateContentConfig{
		SafetySettings: []*genai.SafetySetting{
			{Category: genai.HarmCategoryHarassment, Threshold: genai.HarmBlockThresholdBlockNone},
			{Category: genai.HarmCategoryHateSpeech, Threshold: genai.HarmBlockThresholdBlockNone},
			{Category: genai.HarmCategorySexuallyExplicit, Threshold: genai.HarmBlockThresholdBlockNone},
			{Category: genai.HarmCategoryDangerousContent, Threshold: genai.HarmBlockThresholdBlockNone},
		},
	}

	resp, err := geminiClient.Models.GenerateContent(ctx, modelName, genai.Text(prompt), config)
	if err != nil {
		return "", err
	}
	return cleanModelOutput(resp.Text()), nil
}

func cleanModelOutput(text string) string {
	cleaned := strings.TrimSpace(text)
	cleaned = strings.TrimPrefix(cleaned, "```json")
	cleaned = strings.TrimPrefix(cleaned, "```JSON")
	cleaned = strings.TrimPrefix(cleaned, "```")
	cleaned = strings.TrimSuffix(cleaned, "```")
	return strings.TrimSpace(cleaned)
}

func generateDefaultModelText(ctx context.Context, prompt string) (string, error) {
	return generateModelText(ctx, defaultGeminiModel, prompt)
}

// generateJudgeModelText retries transient overload without extending the caller's deadline.
func generateJudgeModelText(ctx context.Context, prompt string) (string, error) {
	for attempt := 0; ; attempt++ {
		if err := ctx.Err(); err != nil {
			return "", err
		}
		text, err := generateDefaultModelText(ctx, prompt)
		var apiError genai.APIError
		if err == nil || attempt >= 2 || !errors.As(err, &apiError) || apiError.Code != http.StatusServiceUnavailable {
			return text, err
		}

		timer := time.NewTimer(time.Second << attempt)
		select {
		case <-ctx.Done():
			timer.Stop()
			return "", ctx.Err()
		case <-timer.C:
		}
	}
}

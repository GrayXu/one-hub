package openaisb

import (
	"errors"
	"fmt"
	"one-api/common"
	"one-api/model"
	"strconv"
)

func (p *OpenaiSBProvider) Balance(channel *model.Channel) (float64, error) {
	fullRequestURL := p.GetFullRequestURL("/sb-api/user/status", "")
	fullRequestURL = fmt.Sprintf("%s?api_key=%s", fullRequestURL, channel.Key)
	headers := p.GetRequestHeaders()

	client := common.NewClient()
	req, err := client.NewRequest("GET", fullRequestURL, common.WithBody(nil), common.WithHeader(headers))
	if err != nil {
		return 0, err
	}

	// 发送请求
	var response OpenAISBUsageResponse
	err = client.SendRequest(req, &response)
	if err != nil {
		return 0, err
	}

	if response.Data == nil {
		return 0, errors.New(response.Msg)
	}
	balance, err := strconv.ParseFloat(response.Data.Credit, 64)
	if err != nil {
		return 0, err
	}
	channel.UpdateBalance(balance)
	return balance, nil
}
import { expect, test } from '@playwright/test'

test('replays a correction and inspects evidence without overflow', async ({ page }) => {
  await page.goto('/')
  await expect(page.getByText('Chronological replay')).toBeVisible()
  await page.getByRole('button', { name: 'Replay case' }).click()
  await expect(page.getByText('Replay has not started.')).toBeVisible()
  await page.getByLabel('Replay position').fill('3')
  await expect(page.getByText('I actually joined on the 18th', { exact: false }).first()).toBeVisible()
  await page.getByText('I actually joined on the 18th', { exact: false }).first().click()
  await expect(page.getByText('msg_conv_004_002')).toBeVisible()
  const overflow = await page.evaluate(() => ({
    body: document.body.scrollWidth > document.documentElement.clientWidth,
    clipped: [...document.querySelectorAll('button, section, header')].some((node) => {
      const rect = node.getBoundingClientRect()
      return rect.right > document.documentElement.clientWidth + 1 || rect.left < -1
    }),
  }))
  expect(overflow).toEqual({ body: false, clipped: false })
})

test('switches guided cases and mobile panels', async ({ page }, testInfo) => {
  test.skip(testInfo.project.name !== 'mobile', 'mobile-only interaction')
  await page.goto('/')
  await page.getByLabel('Guided case').selectOption('abstention_001')
  await expect(page.getByLabel('Guided case')).toHaveValue('abstention_001')
  await page.getByRole('tab', { name: 'answer' }).click()
  await expect(page.getByText('The history does not establish what subject Maya studied')).toBeVisible()
  await expect(page.getByText('No evidence retrieved. The answer abstains.')).toBeVisible()
  await page.getByRole('tab', { name: 'memory' }).click()
  await expect(page.getByRole('button', { name: /College subject unknown/ })).toBeVisible()
})

test('switches baselines, filters status, and shows an empty state', async ({ page }) => {
  await page.goto('/')
  await page.getByRole('button', { name: 'Scorecard' }).click()
  await expect(page.getByText('openai-gpt41-v1')).toBeVisible()
  await page.getByLabel('Baseline').selectOption('B7')
  await expect(page.getByRole('cell', { name: 'B7 Evidence gated' })).toBeVisible()
  await page.getByLabel('Status').selectOption('pilot_scored')
  await expect(page.getByText('No baselines match this filter.')).toBeVisible()
  await page.getByLabel('Baseline').selectOption('all')
  await expect(page.getByRole('cell', { name: 'B1 Full history' })).toBeVisible()
})

test('renders the API error state', async ({ page }) => {
  await page.route('**/api/demo/cases', (route) => route.fulfill({ status: 500, body: '{}' }))
  await page.goto('/')
  await expect(page.getByRole('heading', { name: 'Demo unavailable' })).toBeVisible()
})

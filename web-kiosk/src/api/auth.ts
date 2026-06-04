import { apiFetch } from './client'

export async function login(username: string, password: string): Promise<void> {
  await apiFetch<{ message: string }>('/api/login', {
    method: 'POST',
    body: JSON.stringify({ username, password }),
  })
}

export async function logout(): Promise<void> {
  await apiFetch<{ message: string }>('/api/logout', { method: 'POST' })
}

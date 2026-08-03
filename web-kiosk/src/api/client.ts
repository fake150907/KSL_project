// 401 발생 시 전역 이벤트 발행 → useAuth에서 로그인 화면으로 리다이렉트
export const AUTH_EXPIRED_EVENT = 'kiosk:auth-expired'

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message)
    this.name = 'ApiError'
  }
}

export async function apiFetch<T>(
  url: string,
  options: RequestInit = {},
): Promise<T> {
  const res = await fetch(url, {
    ...options,
    credentials: 'include',
    headers: {
      'Content-Type': 'application/json',
      ...options.headers,
    },
  })

  if (res.status === 401) {
    window.dispatchEvent(new Event(AUTH_EXPIRED_EVENT))
    throw new ApiError(401, '세션이 만료되었습니다.')
  }

  if (!res.ok) {
    const body = await res.json().catch(() => ({}))
    throw new ApiError(res.status, body?.error ?? res.statusText)
  }

  // 204 No Content
  if (res.status === 204) return undefined as T

  return res.json() as Promise<T>
}

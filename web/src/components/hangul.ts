export const CHO  = ['ㄱ','ㄲ','ㄴ','ㄷ','ㄸ','ㄹ','ㅁ','ㅂ','ㅃ','ㅅ','ㅆ','ㅇ','ㅈ','ㅉ','ㅊ','ㅋ','ㅌ','ㅍ','ㅎ']
export const JUNG = ['ㅏ','ㅐ','ㅑ','ㅒ','ㅓ','ㅔ','ㅕ','ㅖ','ㅗ','ㅘ','ㅙ','ㅚ','ㅛ','ㅜ','ㅝ','ㅞ','ㅟ','ㅠ','ㅡ','ㅢ','ㅣ']
export const JONG = ['','ㄱ','ㄲ','ㄳ','ㄴ','ㄵ','ㄶ','ㄷ','ㄹ','ㄺ','ㄻ','ㄼ','ㄽ','ㄾ','ㄿ','ㅀ','ㅁ','ㅂ','ㅄ','ㅅ','ㅆ','ㅇ','ㅈ','ㅊ','ㅋ','ㅌ','ㅍ','ㅎ']

/** 초성 + 중성 + 종성을 받아 완성형 한글 한 글자를 반환 */
export function assembleHangul(cho: string, jung: string, jong: string): string {
  if (!cho || !jung) return cho || ''
  const ci = CHO.indexOf(cho)
  const ji = JUNG.indexOf(jung)
  const oi = JONG.indexOf(jong)
  if (ci === -1 || ji === -1) return ''
  return String.fromCharCode(0xAC00 + (ci * 21 + ji) * 28 + (oi === -1 ? 0 : oi))
}

/** 숫자 문자열을 010-XXXX-XXXX 형태로 포맷 */
export function formatPhone(val: string): string {
  const d = val.replace(/\D/g, '')
  if (d.length <= 3) return d
  if (d.length <= 7) return `${d.slice(0,3)}-${d.slice(3)}`
  return `${d.slice(0,3)}-${d.slice(3,7)}-${d.slice(7)}`
}

export interface CitizenData {
  name:   string
  dob:    string
  gender: string
  phone:  string
}

// 💡 [수정됨] 'confirm' 단계가 추가되었습니다.
export type Step     = 'start' | 'name' | 'dob' | 'gender' | 'phone' | 'confirm' | 'waiting'
export type CharStep = 'cho' | 'jung' | 'jong'
declare module 'bun:sqlite' {
  export class Statement<Params extends any[] = any[], Result = any> {
    run(...params: Params): any;
    get(...params: Params): Result;
    all(...params: Params): Result[];
    values(...params: Params): any[];
    iterate(...params: Params): IterableIterator<Result>;
    as<T>(): Statement<Params, T>;
    finalize(...params: any[]): void;
  }

  export class Database {
    filename?: string;
    constructor(filename?: string, options?: any);
    prepare<Params extends any[] = any[], Result = any>(query: string): Statement<Params, Result>;
    query<Params extends any[] = any[], Result = any>(query: string): Statement<Params, Result>;
    run(query: string, ...params: any[]): any;
    exec(query: string): void;
    transaction<T extends (...args: any[]) => any>(fn: T): T;
    close(): void;
  }
}
